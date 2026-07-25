#!/usr/bin/env python
from itertools import product
from pathlib import Path
import re
from typing import Annotated, TYPE_CHECKING
if TYPE_CHECKING:
    import torch
import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, Field
import typer
import yaml

from neuralnet.datasets import ParquetDataset


from ..submitit import ExecutorConfig
from ..pydantic import pydantic_to_markdown_schema


def get_model(input_dim: int, grid_size: int, spline_order: int):
    from efficient_kan import KAN

    return KAN([input_dim, 5, 1], grid_size=grid_size, spline_order=spline_order)


def get_n_folds(kfold_table_glob: str, fold_col: str) -> int:
    max_fold = (
        pl.scan_parquet(kfold_table_glob)
        .filter(pl.col(fold_col).is_not_null())
        .select(pl.col(fold_col).max().alias("max_fold"))
        .collect()
        .item()
    )
    return int(max_fold) + 1


def norm1(data):
    norms = np.abs(data.sum(axis=1))
    norms[norms == 0] = 1
    return data / norms[:, None]


def get_ring_indexes() -> list[int]:
    ring_indexes = []
    ring_indexes += list(range(8 // 2))
    sum_rings = 8
    ring_indexes += list(range(sum_rings, sum_rings + (64 // 2)))
    sum_rings = 8 + 64
    ring_indexes += list(range(sum_rings, sum_rings + (8 // 2)))
    sum_rings = 8 + 64 + 8
    ring_indexes += list(range(sum_rings, sum_rings + (8 // 2)))
    sum_rings = 8 + 64 + 8 + 8
    ring_indexes += list(range(sum_rings, sum_rings + (4 // 2)))
    sum_rings = 8 + 64 + 8 + 8 + 4
    ring_indexes += list(range(sum_rings, sum_rings + (4 // 2)))
    sum_rings = 8 + 64 + 8 + 8 + 4 + 4
    ring_indexes += list(range(sum_rings, sum_rings + (4 // 2)))
    return ring_indexes


def get_data(
    ring_indexes: list[int],
    data_table_glob: Path,
    kfold_table_glob: Path,
    fold: int,
    et_bin_left: float,
    et_bin_right: float,
    eta_bin_left: float,
    eta_bin_right: float,
    et_col: str,
    eta_col: str,
    rings_col: str,
    label_col: str,
    fold_col_name: str,
    keep_id: bool = False,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    et = pl.col(et_col)
    et_bin_left_lit = pl.lit(et_bin_left, dtype=pl.dtype_of(et))
    et_bin_right_lit = pl.lit(et_bin_right, dtype=pl.dtype_of(et))

    eta = pl.col(eta_col).abs()
    eta_bin_left_lit = pl.lit(eta_bin_left, dtype=pl.dtype_of(eta))
    eta_bin_right_lit = pl.lit(eta_bin_right, dtype=pl.dtype_of(eta))

    rings = [pl.col(rings_col).list.get(i).alias(f"rings_{i}") for i in ring_indexes]

    data_df = (
        pl.scan_parquet(data_table_glob)
        .filter(
            et.is_between(et_bin_left_lit, et_bin_right_lit, closed="left")
            & eta.is_between(eta_bin_left_lit, eta_bin_right_lit, closed="left")
        )
        .select("id", *rings)
    )

    label = pl.col(label_col)
    fold_col = pl.col(fold_col_name)
    fold_lit = pl.lit(fold, dtype=pl.dtype_of(fold_col))

    val_fold_df = (
        pl.scan_parquet(kfold_table_glob)
        .filter((fold_col == fold_lit) & label.is_not_null())
        .select("id", label.cast(pl.Int32))
    )

    train_fold_df = (
        pl.scan_parquet(kfold_table_glob)
        .filter((fold_col != fold_lit) & label.is_not_null())
        .select("id", label.cast(pl.Int32))
    )

    train_df = data_df.join(train_fold_df, on="id", how="inner")
    val_df = data_df.join(val_fold_df, on="id", how="inner")

    if not keep_id:
        train_df = train_df.drop("id")
        val_df = val_df.drop("id")

    return train_df, val_df

type DeviceType = 'torch.device' | str

def load_trained_model(
    model_path: Path,
    grid_size: int = 5,
    spline_order: int = 3,
    device: DeviceType = "cpu",
) -> 'torch.nn.Module':
    import torch
    if model_path.is_dir():
        model_path = model_path / "model_weights.pth"

    device = torch.device(device)
    state_dict = torch.load(model_path, map_location=device)
    input_dim = state_dict["layers.0.base_weight"].shape[1]

    model = get_model(input_dim, grid_size, spline_order)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def parse_model_dir(
    model_dir: Path,
) -> tuple[tuple[float, float], tuple[float, float], int, int]:
    parts = model_dir.parts

    et_range = None
    eta_range = None
    fold = None
    init = None

    for part in parts:
        if et_range is None and part.startswith("et_"):
            left, right = part.removeprefix("et_").split("_", maxsplit=1)
            et_range = (float(left), float(right))
            continue

        if eta_range is None and part.startswith("eta_"):
            left, right = part.removeprefix("eta_").split("_", maxsplit=1)
            eta_range = (float(left), float(right))
            continue

        fold_init_match = re.fullmatch(r"fold_(\d+)_init_(\d+)", part)
        if fold_init_match is not None:
            fold = int(fold_init_match.group(1))
            init = int(fold_init_match.group(2))

    if et_range is None or eta_range is None or fold is None or init is None:
        raise ValueError(
            "Could not infer et/eta bins and fold from model path. "
            f"Expected path parts like et_* / eta_* / fold_*_init_*. Got: {model_dir}"
        )

    return et_range, eta_range, fold, init


def _discover_model_dirs(model_path: Path) -> list[Path]:
    if model_path.is_dir() and (model_path / "model_weights.pth").is_file():
        return [model_path]

    if model_path.is_dir():
        model_dirs = sorted(
            path for path in model_path.rglob("*.model") if path.is_dir()
        )
        if model_dirs:
            return model_dirs

    if model_path.suffix == ".model" and model_path.is_dir():
        return [model_path]

    raise ValueError(f"No model directories found under: {model_path}")


def _load_val_sp(model_dir: Path) -> float:
    result_file = model_dir / "results.pic"
    if not result_file.is_file():
        raise FileNotFoundError(f"Missing results file: {result_file}")

    with result_file.open("rb") as f:
        results = pd.read_pickle(f)

    return float(results["history"]["val_sp"][-1])


def _select_best_model_dirs(model_dirs: list[Path]) -> list[dict[str, object]]:
    best_by_region: dict[
        tuple[tuple[float, float], tuple[float, float]], dict[str, object]
    ] = {}

    for model_dir in model_dirs:
        et_bin, eta_bin, fold, init = parse_model_dir(model_dir)
        val_sp = _load_val_sp(model_dir)
        region_key = (et_bin, eta_bin)
        current_best = best_by_region.get(region_key)

        candidate = {
            "model_dir": model_dir,
            "et_bin": et_bin,
            "eta_bin": eta_bin,
            "fold": fold,
            "init": init,
            "val_sp": val_sp,
        }

        if current_best is None or val_sp > current_best["val_sp"]:
            best_by_region[region_key] = candidate

    return [best_by_region[key] for key in sorted(best_by_region)]


def _load_val_data_with_metadata(
    fold: int,
    et_bin: tuple[float, float],
    eta_bin: tuple[float, float],
    dataset: ParquetDataset,
    data_table: str,
    kfold_table: str,
    et_col: str,
    eta_col: str,
    rings_col: str,
    label_col: str,
    fold_col: str,
) -> tuple['torch.Tensor', 'pd.DataFrame']:
    import torch
    ring_indexes = get_ring_indexes()
    ring_cols = [f"rings_{i}" for i in ring_indexes]

    _, val_df = get_data(
        ring_indexes=ring_indexes,
        data_table_glob=dataset.get_table_glob(data_table),
        kfold_table_glob=dataset.get_table_glob(kfold_table),
        fold=fold,
        et_bin_left=et_bin[0],
        et_bin_right=et_bin[1],
        eta_bin_left=eta_bin[0],
        eta_bin_right=eta_bin[1],
        et_col=et_col,
        eta_col=eta_col,
        rings_col=rings_col,
        label_col=label_col,
        fold_col_name=fold_col,
        keep_id=True,
    )
    val_df = val_df.collect()
    val_rings = norm1(val_df.select(ring_cols).to_numpy().astype("float32"))
    val_X = torch.as_tensor(val_rings, dtype=torch.float32)

    metadata_df = val_df.select(
        "id", pl.col(label_col).cast(pl.Int32).alias("label")
    ).to_pandas()
    return val_X, metadata_df


def select_models(path: Path) -> list[dict[str, object]]:
    model_dirs = _discover_model_dirs(path)
    return _select_best_model_dirs(model_dirs)


def model_inference(
    model_path: str | Path,
    dataset_dir: Path,
    data_table: str,
    kfold_table: str,
    et_col: str,
    eta_col: str,
    rings_col: str,
    label_col: str = "label",
    fold_col: str = "kfold",
    device: DeviceType = "cpu",
    clear_cuda_cache: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    dataset = ParquetDataset(dataset_dir=dataset_dir)

    def infer_model(
        model_info: dict[str, object], infer_device: 'torch.device'
    ) -> pd.DataFrame:
        import torch
        model_dir = model_info["model_dir"]
        et_bin = model_info["et_bin"]
        eta_bin = model_info["eta_bin"]
        fold = model_info["fold"]

        if not isinstance(model_dir, Path):
            raise TypeError(
                f"Expected Path at model_info['model_dir'], got: {type(model_dir)}"
            )
        if not isinstance(et_bin, tuple) or not isinstance(eta_bin, tuple):
            raise TypeError(
                "Expected tuple bins at model_info['et_bin'] and model_info['eta_bin']"
            )
        if not isinstance(fold, int):
            raise TypeError(f"Expected int at model_info['fold'], got: {type(fold)}")

        model = load_trained_model(model_path=model_dir, device=infer_device)

        val_X, metadata_df = _load_val_data_with_metadata(
            fold=fold,
            et_bin=et_bin,
            eta_bin=eta_bin,
            dataset=dataset,
            data_table=data_table,
            kfold_table=kfold_table,
            et_col=et_col,
            eta_col=eta_col,
            rings_col=rings_col,
            label_col=label_col,
            fold_col=fold_col,
        )
        val_X = val_X.to(infer_device)

        model.eval()
        with torch.inference_mode():
            logits = model(val_X)
            output = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            logits_np = logits.squeeze(1).detach().cpu().numpy()

        output_df = metadata_df.copy()
        output_df["output"] = output
        output_df["logits"] = logits_np
        output_df["fold"] = fold
        output_df["et_bin_left"] = et_bin[0]
        output_df["et_bin_right"] = et_bin[1]
        output_df["eta_bin_left"] = eta_bin[0]
        output_df["eta_bin_right"] = eta_bin[1]

        return output_df[
            [
                "id",
                "output",
                "logits",
                "fold",
                "et_bin_left",
                "et_bin_right",
                "eta_bin_left",
                "eta_bin_right",
            ]
        ]
    
    from ..torch.inference import model_inference as generic_inference
    return generic_inference(
        model_path=model_path,
        select_models=select_models,
        infer_model=infer_model,
        device=device,
        clear_cuda_cache=clear_cuda_cache,
        show_progress=show_progress,
    )


class VKANTrainingJob(BaseModel):
    dataset_dir: Annotated[
        Path, Field(description="Directory containing the parquet dataset")
    ]
    data_table: Annotated[
        str, Field(description="Name of the data table in the parquet dataset")
    ]
    rings_col: Annotated[
        str, Field(description="Name of the rings column in the data table")
    ]
    kfold_table: Annotated[
        str, Field(description="Name of the kfold table in the parquet dataset")
    ]
    label_col: Annotated[
        str, Field(description="Name of the label column in the kfold table")
    ]
    fold_col: Annotated[
        str, Field(description="Name of the fold column in the kfold table")
    ]
    et_col: Annotated[str, Field(description="Name of the et column in the data table")]
    et_bins: Annotated[
        str | tuple[float, ...],
        Field(
            description='et bin to select from the dataset. Should be in the format "et_bin_left,et_bin_right". Example: "20000,40000"'
        ),
    ]
    eta_col: Annotated[
        str, Field(description="Name of the eta column in the data table")
    ]
    eta_bins: Annotated[
        str | tuple[float, ...],
        Field(
            description='eta bin to select from the dataset. Should be in the format "eta_bin_left,eta_bin_right". Example: "0.0,0.8"'
        ),
    ]
    output_dir: Annotated[
        Path | None, Field(description="Directory to save the training results")
    ] = None
    tag: Annotated[str, Field(description="Tag to identify the training run")]
    grid_size: Annotated[int, Field(description="KAN grid size")] = 5
    spline_order: Annotated[int, Field(description="KAN spline order")] = 3
    batch_size: Annotated[int, Field(description="Batch size for training")] = 1024
    inits: Annotated[int, Field(description="Number of initializations")] = 5
    epochs: Annotated[int, Field(description="Maximum number of epochs")] = 5000
    patience: Annotated[int, Field(description="Early stopping patience")] = 25
    dry_run: Annotated[
        bool, Field(description="Perform a dry run without actually training")
    ] = False
    executor_config: Annotated[
        ExecutorConfig,
        Field(
            description="Slurm configuration for running the training job on a Slurm cluster"
        ),
    ]

    def model_post_init(self, context):
        if isinstance(self.et_bins, str):
            self.et_bins = tuple(float(x) for x in self.et_bins.split(","))
        if isinstance(self.eta_bins, str):
            self.eta_bins = tuple(float(x) for x in self.eta_bins.split(","))

        if self.output_dir is None:
            self.output_dir = Path.cwd() / "vkan_training_job"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        return super().model_post_init(context)

    @classmethod
    def from_yaml(cls, yaml_file: Path, **kwargs) -> "VKANTrainingJob":
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        for key, value in kwargs.items():
            data[key] = value
        return cls(**data)

    def get_data(
        self,
        ring_indexes: list[int],
        data_table_glob: Path,
        kfold_table_glob: Path,
        fold: int,
        et_bin_left: float,
        et_bin_right: float,
        eta_bin_left: float,
        eta_bin_right: float,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        return get_data(
            ring_indexes=ring_indexes,
            data_table_glob=data_table_glob,
            kfold_table_glob=kfold_table_glob,
            fold=fold,
            et_bin_left=et_bin_left,
            et_bin_right=et_bin_right,
            eta_bin_left=eta_bin_left,
            eta_bin_right=eta_bin_right,
            et_col=self.et_col,
            eta_col=self.eta_col,
            rings_col=self.rings_col,
            label_col=self.label_col,
            fold_col_name=self.fold_col,
            keep_id=False,
        )

    def load_data(
        self,
        fold: int,
        et_bin_left: float,
        et_bin_right: float,
        eta_bin_left: float,
        eta_bin_right: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rings_indexes = get_ring_indexes()

        dataset = ParquetDataset(dataset_dir=self.dataset_dir)

        train_df, val_df = self.get_data(
            ring_indexes=rings_indexes,
            data_table_glob=dataset.get_table_glob(self.data_table),
            kfold_table_glob=dataset.get_table_glob(self.kfold_table),
            fold=fold,
            et_bin_left=et_bin_left,
            et_bin_right=et_bin_right,
            eta_bin_left=eta_bin_left,
            eta_bin_right=eta_bin_right,
        )
        train_df, val_df = pl.collect_all([train_df, val_df])

        val_label = val_df.drop_in_place(self.label_col).to_numpy().flatten()
        val_rings = norm1(val_df.to_numpy().astype(np.float32))
        del val_df

        train_label = train_df.drop_in_place(self.label_col).to_numpy().flatten()
        train_rings = norm1(train_df.to_numpy().astype(np.float32))
        del train_df

        return (
            train_rings,
            val_rings,
            train_label.astype(np.int32),
            val_label.astype(np.int32),
        )

    def run_training(
        self,
        et_bin_left: float,
        et_bin_right: float,
        eta_bin_left: float,
        eta_bin_right: float,
        fold: int,
        init: int,
    ):
        from neuralnet import get_logger
        from neuralnet.torch import training_torch
        logger = get_logger()
        logger.info(
            f"Loading data for et_bin ({et_bin_left}, {et_bin_right}), eta_bin ({eta_bin_left}, {eta_bin_right}), fold {fold} and init {init}"
        )
        X, X_val, y, y_val = self.load_data(
            fold=fold,
            et_bin_left=et_bin_left,
            et_bin_right=et_bin_right,
            eta_bin_left=eta_bin_left,
            eta_bin_right=eta_bin_right,
        )
        output_dir = (
            self.output_dir
            / f"et_{et_bin_left}_{et_bin_right}"
            / f"eta_{eta_bin_left}_{eta_bin_right}"
            / f"fold_{fold}_init_{init}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        model = get_model(
            input_dim=X.shape[1],
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )
        training_torch(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            sort=fold,
            init=init,
            tag=self.tag,
            model=model,
            output_dir=str(output_dir),
            batch_size=self.batch_size,
            epochs=self.epochs,
            patience=self.patience,
            verbose=True,
            dry_run=self.dry_run,
            et_bin=(et_bin_left, et_bin_right),
            eta_bin=(eta_bin_left, eta_bin_right),
        )
        logger.info(
            f"Training completed for et_bin ({et_bin_left}, {et_bin_right}), eta_bin ({eta_bin_left}, {eta_bin_right}), fold {fold} and init {init}"
        )

    def run(self):
        from neuralnet import get_logger
        logger = get_logger()
        dataset = ParquetDataset(dataset_dir=str(self.dataset_dir))
        n_folds = get_n_folds(
            kfold_table_glob=dataset.get_table_glob(self.kfold_table),
            fold_col=self.fold_col,
        )
        logger.info(f"The dataset has {n_folds} folds.")
        folds_range = range(n_folds)
        inits_range = range(self.inits)
        et_bins_iterator = zip(
            self.et_bins[:-1],
            self.et_bins[1:],
        )
        eta_bins_iterator = zip(
            self.eta_bins[:-1],
            self.eta_bins[1:],
        )
        executor = self.executor_config.get_executor()
        bins_iterator = product(et_bins_iterator, eta_bins_iterator)
        i = 0
        with executor.batch():
            for (et_bin_left, et_bin_right), (
                eta_bin_left,
                eta_bin_right,
            ) in bins_iterator:
                if i > 0 and self.dry_run:
                    logger.info("Dry run enabled, stopping after first bin.")
                    break
                for fold, init in product(folds_range, inits_range):
                    logger.info(
                        f"{i} - Submitting training job for et_bin ({et_bin_left}, {et_bin_right}), eta_bin ({eta_bin_left}, {eta_bin_right}), fold {fold} and init {init}"
                    )
                    executor.submit(
                        self.run_training,
                        et_bin_left,
                        et_bin_right,
                        eta_bin_left,
                        eta_bin_right,
                        fold,
                        init,
                    )
                    i += 1

        logger.info("All training jobs submitted.")


app = typer.Typer(help="Ringer Zero VKAN commands", rich_markup_mode="markdown")


RUN_TRAINING_HELP = "Run VKAN training jobs"


@app.command(
    short_help=RUN_TRAINING_HELP,
    help=f"**{RUN_TRAINING_HELP}**\n\n{pydantic_to_markdown_schema(VKANTrainingJob)}",
)
def run_training(
    config: Annotated[
        Path,
        typer.Option(
            "--config", help="Path to the YAML configuration file for the training job"
        ),
    ],
):
    job = VKANTrainingJob.from_yaml(config)
    job.run()


@app.command()
def add_inference(
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset-dir", help="Directory containing the parquet dataset"),
    ],
    results_dir: Annotated[
        Path,
        typer.Option(
            "--results-dir",
            help="Directory containing the training results to add inference to",
        ),
    ],
    features_table: Annotated[
        str,
        typer.Option(
            "--features-table",
            help="Name of the table containing the features to run inference on",
        ),
    ],
    inference_table: Annotated[
        str,
        typer.Option(
            "--inference-table", help="Name of the table to save the inference results"
        ),
    ],
    kfold_table: Annotated[
        str,
        typer.Option(
            "--kfold-table",
            help="Name of the kfold table used during training and to select validation samples",
        ),
    ] = "standard_binning_kfold",
    eta_col: Annotated[
        str, typer.Option("--eta-col", help="Name of the eta column in the data table")
    ] = "TrigEMClusterContainer.eta",
    et_col: Annotated[
        str, typer.Option("--et-col", help="Name of the et column in the data table")
    ] = "TrigEMClusterContainer.et",
    rings_col: Annotated[
        str,
        typer.Option("--rings-col", help="Name of the rings column in the data table"),
    ] = "TrigEMClusterContainer.ringsE",
    fold_col: Annotated[
        str,
        typer.Option("--fold-col", help="Name of the fold column in the kfold table"),
    ] = "kfold",
    device: Annotated[
        str, typer.Option("--device", help="Torch device to run inference on")
    ] = "cpu",
    clear_cuda_cache: Annotated[
        bool, typer.Option("--clear-cuda-cache", help="Clear CUDA cache between models")
    ] = True,
    show_progress: Annotated[
        bool, typer.Option("--show-progress", help="Show progress bar during inference")
    ] = True,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = model_inference(
        model_path=results_dir,
        dataset_dir=dataset_dir,
        data_table=features_table,
        kfold_table=kfold_table,
        et_col=et_col,
        eta_col=eta_col,
        rings_col=rings_col,
        fold_col=fold_col,
        device=device,
        clear_cuda_cache=clear_cuda_cache,
        show_progress=show_progress,
    )

    parquet_dataset = ParquetDataset(dataset_dir=dataset_dir)
    prediction_pl = pl.from_pandas(prediction_df)
    prediction_pl.write_parquet(
        pl.PartitionBy(
            str(parquet_dataset.get_table_path(inference_table)),
            max_rows_per_file=100_000,
        ),
        compression="snappy",
    )
    return prediction_df
