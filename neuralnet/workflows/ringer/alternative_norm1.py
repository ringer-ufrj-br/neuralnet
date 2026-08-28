"""Workflow for evaluating quantization error of AlternativeNorm1 normalization.

This module provides :class:`AlternativeNorm1Analysis`, a Pydantic-backed
workflow class that evaluates the impact of fixed-point quantization on
the AlternativeNorm1 normalization by computing MSE, MAE, MAPE, and
KL divergence between unquantized normalized rings and fixed-point
quantized normalized rings across a configurable grid of integer and
fractional bit widths.
"""

from functools import cached_property
import json
from pathlib import Path
from itertools import product
from typing import Annotated, Self
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from ...pydantic import YamlBaseModel
from ...datasets import ParquetDataset, DirectoryType
from ...datasets.ringer import (
    RingerParquetDataset,
    DataTableType,
    RingsColType,
    KFoldTableType,
    LabelColType,
    FoldColType,
    DataGroupType,
    EtBinType,
    EtaBinType,
)
from .fields import RingFractionType
from ...utils.polars import RingSlicesPerLayer
from ...normalizers.polars import AlternativeNorm1
from ... import get_logger
from .training import EtColType, EtaColType, EtBinType, EtaBinType, ExecutorConfigType


RESULTS_COL_DESCRIPTIONS: dict[str, str] = {
    "integer_bits": "Number of integer bits used in fixed-point representation",
    "fractional_bits": "Number of fractional bits used in fixed-point representation",
    "total_bits": "Total bitwidth ($ib + fb$)",
    "mse": "Mean Squared Error: $\\frac{1}{ND}\\sum_{i=1}^N \\sum_{j=1}^D (y_{ij} - \\hat{y}_{ij})^2$",
    "rmse": "Root Mean Squared Error: $\\sqrt{\\text{MSE}}$",
    "mae": "Mean Absolute Error: $\\frac{1}{ND}\\sum_{i=1}^N \\sum_{j=1}^D |y_{ij} - \\hat{y}_{ij}|$",
    "mape": "Mean Absolute Percentage Error in relation to the floating-point normalized rings: $\\frac{1}{ND}\\sum_{i=1}^N \\sum_{j=1}^D \\frac{|y_{ij} - \\hat{y}_{ij}|}{|y_{ij}|}$ (with $\\epsilon$ floor when $|y_{ij}| \\le \\epsilon$)",
    "max_error": "Maximum absolute difference over all events and rings: $\\max_{i,j} |y_{ij} - \\hat{y}_{ij}|$",
    "kl_divergence": "Sample-wise discrete spatial profile KL divergence averaged across all events: $\\frac{1}{N}\\sum_{i=1}^N D_{\\text{KL}}(P_i \\parallel Q_i)$",
    "hist_kl_divergence": "Empirical value distribution KL divergence over 1D value histograms of unquantized vs. quantized rings",
}


def format_dataframe_schema_markdown(
    schema: pl.Schema,
    descriptions: dict[str, str] | None = None,
) -> str:
    """Format a Polars Schema into a Markdown table with column names, data types, and descriptions.

    Parameters
    ----------
    schema : pl.Schema
        Polars schema mapping column names to data types.
    descriptions : dict[str, str] | None, optional
        Optional dictionary of custom descriptions per column name.

    Returns
    -------
    str
        Formatted markdown table string.
    """
    lines = [
        "| Column Name | Data Type | Description |",
        "| :--- | :--- | :--- |",
    ]
    for col_name, dtype in schema.items():
        desc = ""
        if descriptions and col_name in descriptions:
            desc = descriptions[col_name]
        elif col_name.endswith(".diff"):
            base = col_name[:-5]
            desc = f"Signed difference ($y - \\hat{{y}}$) for `{base}`"
        elif col_name.endswith(".abs_diff"):
            base = col_name[:-9]
            desc = f"Absolute difference ($|y - \\hat{{y}}|$) for `{base}`"
        elif col_name.endswith(".sq_diff"):
            base = col_name[:-8]
            desc = f"Squared difference ($(y - \\hat{{y}})^2$) for `{base}`"
        elif col_name.endswith(".rel_diff"):
            base = col_name[:-9]
            desc = f"Relative difference in relation to floating-point normalization ($\\frac{{|y - \\hat{{y}}|}}{{|y|}}$) for `{base}`"
        elif col_name == "id":
            desc = "Event / sample ID"
        elif col_name == "sample_mse":
            desc = "Event-level Mean Squared Error across rings"
        elif col_name == "sample_mae":
            desc = "Event-level Mean Absolute Error across rings"
        elif col_name == "sample_mape":
            desc = "Event-level Mean Absolute Percentage Error in relation to floating-point rings"
        elif col_name == "sample_max_error":
            desc = "Event-level Maximum Absolute Error across rings"
        elif col_name == "sample_kl_divergence":
            desc = (
                "Event-level discrete spatial energy distribution KL divergence ($D_{\\text{KL}}(P_i \\parallel Q_i)$)"
            )
        else:
            desc = "Feature column"
        lines.append(f"| `{col_name}` | `{dtype}` | {desc} |")
    return "\n".join(lines)


class IntegerRange(BaseModel):
    """Integer range defined by start and stop values.

    Attributes
    ----------
    start : int
        Start value of the range (inclusive).
    stop : int
        Stop value of the range (exclusive).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: Annotated[int, Field(ge=0, description="Start value of the range (inclusive).")]
    stop: Annotated[int, Field(ge=0, description="Stop value of the range (exclusive).")]

    def as_range(self) -> range:
        """Return Python range object."""
        return range(self.start, self.stop)

    def as_list(self) -> list[int]:
        """Return range values as a list."""
        return list(range(self.start, self.stop))

    def __iter__(self):
        return iter(range(self.start, self.stop))


def compute_norm_metrics(
    df: pl.DataFrame | pl.LazyFrame,
    norm_cols: list[str],
    quant_cols: list[str],
    eps: float = 1e-12,
    hist_bins: int = 100,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Compute error and distribution distance metrics using Polars expressions.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Polars DataFrame or LazyFrame containing both unquantized and quantized columns.
    norm_cols : list[str]
        Column names of true (unquantized) normalized rings.
    quant_cols : list[str]
        Column names of quantized normalized rings.
    eps : float, default=1e-12
        Small epsilon value for numerical stability in division and log.
    hist_bins : int, default=100
        Number of bins for empirical value distribution histograms.

    Returns
    -------
    tuple[pl.LazyFrame, pl.LazyFrame]
        A tuple consisting of:
        - LazyFrame with one row containing aggregated metrics: mse, rmse, mae, mape, max_error, kl_divergence, and hist_kl_divergence.
        - LazyFrame containing sample-wise differences, absolute differences, squared differences,
          relative differences, and sample-level summary metrics.
    """
    n_features = len(norm_cols)
    df_lazy = df.lazy() if isinstance(df, pl.DataFrame) else df
    schema = df_lazy.collect_schema()

    diff_exprs = [(pl.col(c) - pl.col(qc)).alias(f"{c}.diff") for c, qc in zip(norm_cols, quant_cols)]
    abs_diff_exprs = [(pl.col(c) - pl.col(qc)).abs().alias(f"{c}.abs_diff") for c, qc in zip(norm_cols, quant_cols)]
    diff_sq_exprs = [(pl.col(c) - pl.col(qc)).pow(2).alias(f"{c}.sq_diff") for c, qc in zip(norm_cols, quant_cols)]
    rel_diff_exprs = [
        (pl.col(c) - pl.col(qc))
        .abs()
        .truediv(pl.when(pl.col(c).abs() > eps).then(pl.col(c).abs()).otherwise(eps))
        .alias(f"{c}.rel_diff")
        for c, qc in zip(norm_cols, quant_cols)
    ]

    raw_abs_diffs = [(pl.col(c) - pl.col(qc)).abs() for c, qc in zip(norm_cols, quant_cols)]
    raw_sq_diffs = [(pl.col(c) - pl.col(qc)).pow(2) for c, qc in zip(norm_cols, quant_cols)]
    raw_rel_diffs = [
        (pl.col(c) - pl.col(qc)).abs().truediv(pl.when(pl.col(c).abs() > eps).then(pl.col(c).abs()).otherwise(eps))
        for c, qc in zip(norm_cols, quant_cols)
    ]

    # Sample-wise KL divergence: discrete spatial distribution across rings per event
    # p_raw = [pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(0.0) + eps for c in norm_cols]
    # q_raw = [pl.when(pl.col(qc) > 0).then(pl.col(qc)).otherwise(0.0) + eps for qc in quant_cols]
    # p_sum = pl.sum_horizontal(*p_raw)
    # q_sum = pl.sum_horizontal(*q_raw)
    # kl_terms = [(p_k / p_sum) * ((p_k / p_sum) / (q_k / q_sum)).log() for p_k, q_k in zip(p_raw, q_raw)]

    sample_summary_exprs = [
        pl.sum_horizontal(*raw_sq_diffs).truediv(n_features).alias("sample_mse"),
        pl.sum_horizontal(*raw_abs_diffs).truediv(n_features).alias("sample_mae"),
        pl.sum_horizontal(*raw_rel_diffs).truediv(n_features).alias("sample_mape"),
        pl.max_horizontal(*raw_abs_diffs).alias("sample_max_error"),
        # pl.sum_horizontal(*kl_terms).alias("sample_kl_divergence"),
    ]

    id_exprs = [pl.col("id")] if "id" in schema else []

    sample_diffs_df = df_lazy.select(
        "id",
        *id_exprs,
        *sample_summary_exprs,
        *diff_exprs,
        *abs_diff_exprs,
        *diff_sq_exprs,
        *rel_diff_exprs,
    )

    # bin_edges = np.linspace(0.0, 1.0, hist_bins + 1)
    # p_counts = []
    # q_counts = []
    # for i in range(hist_bins):
    #     low, high = float(bin_edges[i]), float(bin_edges[i + 1])
    #     if i == hist_bins - 1:
    #         p_c = pl.sum_horizontal(
    #             [pl.when((pl.col(c) >= low) & (pl.col(c) <= high)).then(1.0).otherwise(0.0) for c in norm_cols]
    #         ).sum()
    #         q_c = pl.sum_horizontal(
    #             [pl.when((pl.col(c) >= low) & (pl.col(c) <= high)).then(1.0).otherwise(0.0) for c in quant_cols]
    #         ).sum()
    #     else:
    #         p_c = pl.sum_horizontal(
    #             [pl.when((pl.col(c) >= low) & (pl.col(c) < high)).then(1.0).otherwise(0.0) for c in norm_cols]
    #         ).sum()
    #         q_c = pl.sum_horizontal(
    #             [pl.when((pl.col(c) >= low) & (pl.col(c) < high)).then(1.0).otherwise(0.0) for c in quant_cols]
    #         ).sum()
    #     p_counts.append(p_c)
    #     q_counts.append(q_c)

    # p_sum_expr = pl.sum_horizontal(*[c + eps for c in p_counts])
    # q_sum_expr = pl.sum_horizontal(*[c + eps for c in q_counts])

    # kl_bin_terms = []
    # for pc, qc in zip(p_counts, q_counts):
    #     p_d = (pc + eps) / p_sum_expr
    #     q_d = (qc + eps) / q_sum_expr
    #     kl_bin_terms.append(p_d * (p_d / q_d).log())

    # hist_kl_expr = pl.sum_horizontal(*kl_bin_terms).alias("hist_kl_divergence")

    metrics_df = df_lazy.select(
        pl.sum_horizontal(*raw_sq_diffs).truediv(n_features).mean().alias("mse"),
        pl.sum_horizontal(*raw_sq_diffs).truediv(n_features).mean().sqrt().alias("rmse"),
        pl.sum_horizontal(*raw_abs_diffs).truediv(n_features).mean().alias("mae"),
        pl.sum_horizontal(*raw_rel_diffs).truediv(n_features).mean().alias("mape"),
        pl.max_horizontal(*raw_abs_diffs).max().alias("max_error"),
        # pl.sum_horizontal(*kl_terms).mean().alias("kl_divergence"),
        # hist_kl_expr,
    )

    return metrics_df, sample_diffs_df


class AlternativeNorm1Analysis(YamlBaseModel):
    """Workflow to analyze fixed-point quantization error of AlternativeNorm1.

    Compares unquantized normalized rings against fixed-point quantized normalized rings
    across a grid of integer and fractional bit widths, computing MSE, MAE, MAPE,
    and KL divergence metrics, and outputting both aggregated results and sample-wise differences.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Dataset params
    dataset_dir: DirectoryType
    data_table: DataTableType
    rings_col: RingsColType = "rings"
    ring_fraction: RingFractionType = 2

    # Optional split / filtering params
    kfold_table: KFoldTableType | None = None
    label_col: LabelColType = "label"
    fold_col: FoldColType = "kfold"
    et_col: EtColType = "et"
    eta_col: EtaColType = "eta"
    et_bins: EtBinType | None = None
    eta_bins: EtaBinType | None = None

    # Quantization bit ranges
    integer_bits_range: Annotated[
        IntegerRange,
        Field(description="Range of integer bits with start and stop attributes."),
    ]
    fractional_bits_range: Annotated[
        IntegerRange,
        Field(description="Range of fractional bits with start and stop attributes."),
    ]

    # Metric computation params
    eps: Annotated[float, Field(gt=0, description="Epsilon for numerical stability.")] = 1e-12
    hist_bins: Annotated[int, Field(gt=0, description="Number of histogram bins for value distribution KL.")] = 100

    # Execution related fields
    output_path: Annotated[
        Path,
        Field(
            description="Directory path to save results.csv, config.json, README.md, and sample-wise differences parquet files."
        ),
    ]
    # executor_config: ExecutorConfigType

    def get_data(self) -> pl.LazyFrame:
        """Load and filter the dataset as a LazyFrame.

        Returns
        -------
        pl.LazyFrame
            LazyFrame containing data matching configuration filters.
        """

        dataset = ParquetDataset(dataset_dir=self.dataset_dir)
        df = dataset.get_dataframe(self.data_table)

        if self.kfold_table is not None:
            kfold_df = dataset.get_dataframe(self.kfold_table)
            df = df.join(kfold_df, on="id", how="left")

        return df

    def generate_readme_content(
        self,
        results_df: pl.DataFrame | pl.LazyFrame | None = None,
        sample_diffs_df: pl.DataFrame | pl.LazyFrame | None = None,
        n_samples: int | None = None,
        n_features: int | None = None,
    ) -> str:
        """Generate a comprehensive README.md describing the analysis and directory contents.

        Parameters
        ----------
        results_df : pl.DataFrame | pl.LazyFrame | None, optional
            Aggregated results DataFrame with schema.
        sample_diffs_df : pl.DataFrame | pl.LazyFrame | None, optional
            Sample differences DataFrame or LazyFrame with schema.
        n_samples : int | None, optional
            Number of evaluated samples.
        n_features : int | None, optional
            Number of ring features per sample.

        Returns
        -------
        str
            Markdown text content for README.md.
        """
        int_bits = self.integer_bits_range.as_list()
        frac_bits = self.fractional_bits_range.as_list()

        samples_info = f"- **Number of Samples**: {n_samples:,}\n" if n_samples is not None else ""
        features_info = f"- **Ring Features per Sample**: {n_features}\n" if n_features is not None else ""
        kfold_info = (
            f"- **K-Fold Table**: `{self.kfold_table}`\n- **Fold**: {self.fold}\n- **Data Split**: `{self.data_group}`\n"
            if self.kfold_table
            else "- **K-Fold Splitting**: None (entire table used)\n"
        )
        et_bin_info = f"- **ET Bin**: {self.et_bin}\n" if self.et_bin else ""
        eta_bin_info = f"- **Eta Bin**: {self.eta_bin}\n" if self.eta_bin else ""

        first_ib = int_bits[0] if int_bits else 1
        first_fb = frac_bits[0] if frac_bits else 4

        # Generate schema markdown tables
        if results_df is not None:
            results_schema = results_df.collect_schema() if isinstance(results_df, pl.LazyFrame) else results_df.schema
            results_schema_md = format_dataframe_schema_markdown(results_schema, RESULTS_COL_DESCRIPTIONS)
        else:
            default_results_schema = pl.Schema(
                {
                    "integer_bits": pl.Int64,
                    "fractional_bits": pl.Int64,
                    "total_bits": pl.Int64,
                    "mse": pl.Float64,
                    "rmse": pl.Float64,
                    "mae": pl.Float64,
                    "mape": pl.Float64,
                    "max_error": pl.Float64,
                    # "kl_divergence": pl.Float64,
                    # "hist_kl_divergence": pl.Float64,
                }
            )
            results_schema_md = format_dataframe_schema_markdown(default_results_schema, RESULTS_COL_DESCRIPTIONS)

        if sample_diffs_df is not None:
            sample_schema = (
                sample_diffs_df.collect_schema()
                if isinstance(sample_diffs_df, pl.LazyFrame)
                else sample_diffs_df.schema
            )
            sample_diffs_schema_md = format_dataframe_schema_markdown(sample_schema)
        else:
            sample_diffs_schema_md = "| Column Name | Data Type | Description |\n| :--- | :--- | :--- |\n| `id` | `Int64` | Event / sample ID |\n| `sample_mse` | `Float64` | Event-level Mean Squared Error across rings |\n| `sample_mae` | `Float64` | Event-level Mean Absolute Error across rings |\n| `sample_mape` | `Float64` | Event-level Mean Absolute Percentage Error in relation to floating-point rings |\n| `sample_max_error` | `Float64` | Event-level Maximum Absolute Error across rings |\n| `sample_kl_divergence` | `Float64` | Event-level discrete spatial energy distribution KL divergence ($D_{\\text{KL}}(P_i \\parallel Q_i)$) |\n| `<col_name>.diff` | `Float64` | Signed difference ($y - \\hat{y}$) |\n| `<col_name>.abs_diff` | `Float64` | Absolute difference ($|y - \\hat{y}|$) |\n| `<col_name>.sq_diff` | `Float64` | Squared difference ($(y - \\hat{y})^2$) |\n| `<col_name>.rel_diff` | `Float64` | Relative difference in relation to floating-point normalization ($\\frac{|y - \\hat{y}|}{|y|}$) |"

        return f"""# AlternativeNorm1 Fixed-Point Quantization Analysis

This directory contains the results and dataset artifacts generated by the `AlternativeNorm1Analysis` workflow from the `neuralnet` package.

## 1. Overview & Objective

The **AlternativeNorm1** normalization normalizes detector ring energy profiles per event by dividing each ring energy by the absolute value of the sum of ring energies across all rings in that event (unlike the standard $L_1$ norm which divides by the sum of absolute values):

$$\\hat{{x}}_i = \\frac{{x_i}}{{\\left|\\sum_{{j=1}}^D x_j\\right|}} \\quad \\left(\\text{{or }} x_i \\text{{ if }} \\left|\\sum_{{j=1}}^D x_j\\right| = 0\\right)$$

This analysis evaluates the impact of fixed-point arithmetic quantization on the normalized features across a grid of **integer bit widths** and **fractional bit widths**. The evaluation includes error metrics (MSE, RMSE, MAE, MAPE, Max Error) and distribution divergence measures (sample-wise spatial profile KL divergence and value-histogram KL divergence).

> **Note on Relative Differences**: All relative differences and MAPE values are computed in relation to the unquantized floating-point normalized values $y$ (i.e. $\\frac{{|y - \\hat{{y}}|}}{{|y|}}$). A small epsilon floor ($\\epsilon = {self.eps}$) is applied only when $|y| \\le \\epsilon$ to ensure numerical stability without altering the reference frame.

---

## 2. Experiment & Dataset Configuration

- **Dataset Directory**: `{self.dataset_dir}`
- **Source Table**: `{self.data_table}`
{kfold_info}- **Rings Column**: `{self.rings_col}`
- **Ring Slices Fraction**: `{self.ring_fraction}`
{et_bin_info}{eta_bin_info}{samples_info}{features_info}- **Integer Bits Range**: `[{self.integer_bits_range.start}, {self.integer_bits_range.stop})` -> {int_bits}
- **Fractional Bits Range**: `[{self.fractional_bits_range.start}, {self.fractional_bits_range.stop})` -> {frac_bits}
- **Total Tested Bit Configurations**: {len(int_bits) * len(frac_bits)}
- **Numerical Stability Epsilon**: `{self.eps}`
- **Histogram Bins**: `{self.hist_bins}`

---

## 3. Directory Layout & File Descriptions

```
.
├── README.md               # This documentation file
├── config.json             # Full JSON configuration used to execute the analysis
├── results.csv             # Summary table with aggregated metrics across all bit configurations
└── differences/            # Subdirectory containing sample-wise differences per bit configuration
    ├── differences_ib_{first_ib}_fb_{first_fb}.parquet
    └── ...
```

### 3.1. `config.json`
Contains the complete serializable configuration of the `AlternativeNorm1Analysis` job for exact reproducibility.

### 3.2. `results.csv` Schema
A CSV table where each row corresponds to one `(integer_bits, fractional_bits)` configuration:

{results_schema_md}

### 3.3. `differences/differences_ib_{first_ib}_fb_{first_fb}.parquet` Schema
Each Parquet file in the `differences/` directory stores sample-wise and feature-wise evaluation for that specific bitwidth:

{sample_diffs_schema_md}

---

## 4. How to Use & Inspect in Python

### Using `neuralnet` (Recommended)
```python
import polars as pl
from neuralnet.workflows.ringer.alternative_norm1 import AlternativeNorm1Analysis

# 1. Load the analysis instance from this directory
analysis = AlternativeNorm1Analysis.load(".")

# 2. Inspect the aggregated metrics
results_df = analysis.results.collect()
print(results_df)

# 3. Retrieve sample-wise differences for a specific configuration (e.g., {first_ib} integer bit, {first_fb} fractional bits)
diff_df = analysis.get_quantized_df(integer_bits={first_ib}, fractional_bits={first_fb}).collect()
print(diff_df)
```

### Using Standard Polars / Pandas
```python
import polars as pl

# Read summary metrics
results = pl.read_csv("results.csv")

# Read per-sample differences
diffs = pl.read_parquet("differences/differences_ib_{first_ib}_fb_{first_fb}.parquet")
```
"""

    def submit(self) -> pl.DataFrame:
        """Run the quantization analysis and return the metrics DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame containing integer_bits, fractional_bits, total_bits, mse, rmse, mae, mape,
            max_error, kl_divergence, and hist_kl_divergence.
        """
        logger = get_logger()
        logger.info(f"Starting AlternativeNorm1Analysis on table {self.data_table}")

        data_df = self.get_data()

        ring_selector = RingSlicesPerLayer(
            rings_col=self.rings_col,
            fraction=self.ring_fraction,
            output_format="expanded_columns",
        )
        sliced_df = data_df.pipe(ring_selector, passthrough=True)

        normalizer = AlternativeNorm1(input_cols=ring_selector.output_cols)
        norm_df = sliced_df.pipe(normalizer, passthrough=True)

        norm_cols = normalizer.output_cols
        n_features = len(norm_cols)
        logger.info(f"Normalized data with {n_features} ring features")

        int_bits_list = self.integer_bits_range.as_list()
        frac_bits_list = self.fractional_bits_range.as_list()

        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.quantized_dir.mkdir(parents=True, exist_ok=True)

        output_columns = {"id"} | set(normalizer.output_cols)
        # if self.kfold_col:
        #     output_columns.add(self.kfold_col)
        if self.et_col:
            output_columns.add(self.et_col)
        if self.eta_col:
            output_columns.add(self.eta_col)
        if self.label_col:
            output_columns.add(self.label_col)

        all_metric_dfs = []
        for ib, fb in product(int_bits_list, frac_bits_list):
            logger.info(f"Computing quantization for {ib} integer bits and {fb} fractional bits...")
            quantized_normalizer = normalizer.fixed_point_quantization(
                {"fractional_bits": fb, "integer_bits": ib}
            ).model_copy(update={"input_cols": ring_selector.output_cols, "suffix": "quantized_alternative_norm1"})
            quantized_df = norm_df.pipe(quantized_normalizer, passthrough=True)
            iteration_output_columns = output_columns | set(quantized_normalizer.output_cols)

            quantized_df = quantized_df.select(*iteration_output_columns)
            quantized_path = str(self.get_quantized_df_path(ib, fb))
            quantized_df.sink_parquet(quantized_path)
            quantized_df = pl.scan_parquet(quantized_path)

            quantized_metric_dfs = []

            for et_bin, eta_bin in product(self.et_bins, self.eta_bins):
                binned_df = quantized_df.filter(
                    et_bin.as_polars_expr(self.et_col) & eta_bin.as_polars_expr(self.eta_col)
                )
                metrics_lf, _ = compute_norm_metrics(
                    df=binned_df,
                    norm_cols=norm_cols,
                    quant_cols=quantized_normalizer.output_cols,
                    eps=self.eps,
                    hist_bins=self.hist_bins,
                )

                cfg_metrics_lf = metrics_lf.with_columns(
                    pl.lit(ib).alias("integer_bits"),
                    pl.lit(fb).alias("fractional_bits"),
                    et_bin.as_polars_struct().alias("et_bin"),
                    eta_bin.as_polars_struct().alias("eta_bin"),
                ).select(
                    "integer_bits",
                    "fractional_bits",
                    # "data_group",
                    "et_bin",
                    "eta_bin",
                    "mse",
                    "rmse",
                    "mae",
                    "mape",
                    "max_error",
                    # "kl_divergence",
                    # "hist_kl_divergence",
                )
                quantized_metric_dfs.append(cfg_metrics_lf)

            logger.info(f"Aggregating and collecting metrics for IB {ib} and FB {fb}...")
            all_metric_dfs.append(pl.concat(quantized_metric_dfs).collect())

        logger.info("Collecting results_table")
        results_df = pl.concat(all_metric_dfs)

        if self.output_path is not None:
            output_dir = Path(self.output_path)
            results_df.write_csv(output_dir / "results.csv")
            output_dir.joinpath("config.json").write_text(self.model_dump_json(indent=4))
            readme_content = self.generate_readme_content(
                results_df=results_df,
                sample_diffs_df=quantized_df,
                n_samples=quantized_df.select(pl.len()).collect().item(),
                n_features=n_features,
            )
            output_dir.joinpath("README.md").write_text(readme_content)
            logger.info(f"Saved aggregated results, sample differences, config.json, and README.md to {output_dir}")

        return results_df

    def compute_for_bits(self, integer_bits: int, fractional_bits: int):
        logger = get_logger()
        logger.info(f"Computing quantization for {integer_bits} integer bits and {fractional_bits} fractional bits...")
        quantized_normalizer = normalizer.fixed_point_quantization(
            {"fractional_bits": fractional_bits, "integer_bits": integer_bits}
        ).model_copy(update={"input_cols": ring_selector.output_cols, "suffix": "quantized_alternative_norm1"})
        quantized_df = norm_df.pipe(quantized_normalizer, passthrough=True)
        iteration_output_columns = output_columns | set(quantized_normalizer.output_cols)

        quantized_df = quantized_df.select(*iteration_output_columns)
        quantized_path = str(self.get_quantized_df_path(integer_bits, integer_bits))
        quantized_df.sink_parquet(quantized_path)
        quantized_df = pl.scan_parquet(quantized_path)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load an AlternativeNorm1Analysis instance from an output directory.

        Parameters
        ----------
        path : Path | str
            Directory path where analysis results and config.json are saved.

        Returns
        -------
        Self
            Loaded AlternativeNorm1Analysis instance.
        """
        cls.validate_saved_directory(path)
        path = Path(path)
        config_path = path / "config.json"
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        config["output_path"] = path
        instance = cls(**config)
        return instance

    @cached_property
    def quantized_dir(self) -> Path:
        return Path(self.output_path) / "quantized"

    def get_quantized_df_path(
        self,
        integer_bits: int,
        fractional_bits: int,
    ) -> Path:
        """Get the path to the sample-wise differences parquet file for a specific bit configuration.

        Parameters
        ----------
        integer_bits : int
            Number of integer bits.
        fractional_bits : int
            Number of fractional bits.

        Returns
        -------
        Path
            Path to the corresponding differences parquet file.
        """
        return self.quantized_dir / f"quantized_ib_{integer_bits:04d}_fb_{fractional_bits:04d}.parquet"

    @staticmethod
    def validate_saved_directory(output_path: Path | str) -> None:
        """Validate that an output directory exists and contains expected files.

        Parameters
        ----------
        output_path : Path | str
            Directory path to validate.
        """
        output_path = Path(output_path)
        if not output_path.exists():
            raise FileNotFoundError(f"Path '{output_path}' does not exist.")
        if not output_path.is_dir():
            raise NotADirectoryError(f"Path '{output_path}' is not a directory.")
        if not (output_path / "config.json").exists():
            raise FileNotFoundError(f"Config file not found in '{output_path}'.")
        if not (output_path / "results.csv").exists():
            raise FileNotFoundError(f"Results file 'results.csv' not found in '{output_path}'.")

    def get_intermediate_results_path(self, integer_bits: int, floating_bits: int):
        return self.quantized_dir / f"results_ib_{integer_bits:04d}_fb_{floating_bits:04d}.csv"

    @property
    def results(self) -> pl.LazyFrame:
        """Return a Polars LazyFrame of the aggregated quantization results from results.csv.

        Returns
        -------
        pl.LazyFrame
            LazyFrame reading results.csv.
        """
        if self.output_path is None:
            raise ValueError("output_path is not set for this analysis.")
        results_path = Path(self.output_path) / "results.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found at '{results_path}'.")
        return pl.scan_csv(results_path)

    def get_quantized_df(
        self,
        integer_bits: int,
        fractional_bits: int,
    ) -> pl.LazyFrame:
        """Get the sample-wise differences dataframe for a specific bit configuration.

        Parameters
        ----------
        integer_bits : int
            Number of integer bits.
        fractional_bits : int
            Number of fractional bits.

        Returns
        -------
        pl.LazyFrame
            LazyFrame reading the corresponding differences parquet file.
        """
        diff_path = self.get_quantized_df_path(integer_bits, fractional_bits)
        if not diff_path.exists():
            raise FileNotFoundError(f"Differences file not found at '{diff_path}'.")
        return pl.scan_parquet(diff_path)
