"""Ringer committee threshold-fit job.

This module provides :class:`RingerCommitteeThresholdFitJob`, a Pydantic-backed
workflow class that evaluates a trained Ringer specialist committee by fitting
decision thresholds at configurable reference operating points.

Overview
--------
The job loads a completed :class:`~neuralnet.workflows.ringer.training.RingerKerasTrainingJob`
output directory, runs inference on a (potentially different) evaluation
dataset, and — for every specialist model, every dataset split, and every named
reference operating point — locates the decision threshold that best matches
the target true-positive rate (TPR).

Outputs include per-member confusion matrix DataFrames (saved as parquet),
a flat results table, committee-level aggregate metrics, and standalone
operating-point JSON configuration files (e.g. ``tight.json``, ``medium.json``)
alongside a ``models/`` directory with per-member model files and evaluation
outputs. Each operating-point JSON file contains everything needed to
reconstruct the complete inference pipeline from scratch without relying on
the training job directory.

Output directory layout
-----------------------
After :meth:`RingerCommitteeThresholdFitJob.submit` completes, the
``output_path`` directory contains the following files and subdirectories::

    output_path/
    ├── config.json
    │   Job configuration dumped as JSON for reproducibility.
    │
    ├── models/
    │   Directory containing all specialist member directories:
    │   └── member_{id}/
    │       ├── model.keras
    │       │   Full Keras model architecture and trained weights for this
    │       │   committee member.
    │       └── {dataset_type}_confusion_matrix.parquet
    │           Confusion matrix DataFrame containing TPR, FPR, thresholds, and
    │           counts for that dataset split. One file per entry in ``dataset_types``.
    │
    ├── {op_point}.json (e.g. tight.json, medium.json, ...)
    │   One standalone JSON file per reference operating point defined in
    │   ``references``.  Contains all preprocessing pipeline parameters, bin
    │   definitions, relative paths to the model files in ``models/``, and the
    │   fitted decision threshold for each specialist member.  Can be passed
    │   directly to :meth:`RingerCommitteeThresholdFitJob.get_specialist_committee`
    │   or :meth:`~neuralnet.workflows.ringer.models.BinnedSpecialistCommittee.from_json`
    │   to build the complete committee pipeline from scratch.  Structure::
    │
    │       {
    │           "op_point": "tight",
    │           "reference": {"tpr": 0.995, "color": "red", "label": "Tight"},
    │           "preprocessing": {
    │               "rings_col": "rings",
    │               "ring_fraction": 2,
    │               "norm_strategy": "l1"
    │           },
    │           "dataset_types": ["train", "val", "test"],
    │           "models": [
    │               {
    │                   "id": 0,
    │                   "fold": 0,
    │                   "model_path": "models/member_0/model.keras",
    │                   "decision_threshold": 0.8523,
    │                   "bins": [
    │                       {
    │                           "type": "VariableBin",
    │                           "var_name": "et",
    │                           "low": 0.0,
    │                           "high": 100000.0,
    │                           "closed": "left"
    │                       },
    │                       {
    │                           "type": "AbsoluteVariableBin",
    │                           "var_name": "eta",
    │                           "low": 0.0,
    │                           "high": 2.5,
    │                           "closed": "left"
    │                       }
    │                   ],
    │                   "preprocessing": {
    │                       "rings_col": "rings",
    │                       "ring_fraction": 2,
    │                       "norm_strategy": "l1"
    │                   },
    │                   "threshold_fit_results": {
    │                       "train": {"threshold": 0.852, "tpr": 0.995, "fpr": 0.02},
    │                       "val": {"threshold": 0.851, "tpr": 0.995, "fpr": 0.021},
    │                       "test": {"threshold": 0.8523, "tpr": 0.9951, "fpr": 0.023}
    │                   },
    │                   "training_results": { ... }
    │               },
    │               ...
    │           ]
    │       }
    │
    ├── results.parquet
    │   One row per specialist member.  Columns are a flat (dot-separated)
    │   expansion of each member's training results merged with the
    │   threshold-fit outcomes for every (dataset_type, reference) pair.
    │   Key column groups:
    │     • id, fold, et_bin.*, eta_bin.*  — member identity and binning
    │     • {dataset_type}.{ref}.threshold — fitted decision threshold
    │     • {dataset_type}.{ref}.tpr / .fpr / .tp / .tn / .fp / .fn
    │       / .positives / .negatives / .total / .correct / .incorrect
    │                                   — confusion-matrix statistics at the
    │                                     fitted threshold for that split
    │
    └── committee_results.json
        Committee-level aggregate metrics.  Confusion-matrix counts are
        *summed* across all specialists for each (dataset_type, reference)
        pair; derived rates (tpr, fpr, accuracy, sp) are then computed from
        those totals.

"""

import json
import logging
import shutil
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import Annotated, Literal, Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

from ...bins import AbsoluteVariableBin
from ...datasets import DirectoryType
from ...datasets.ringer import (
    DataTableType,
    FoldColType,
    KFoldTableType,
    LabelColType,
    RingsColType,
)
from ...json import cast_to_json_value
from ...logging import LoggerName
from ...metrics import enhanced_confusion_matrix_from_polars, polars_sp_index
from ...pydantic import YamlBaseModel
from ...utils.polars import unnest_structs
from .models import BinnedSpecialistCommittee
from .training import EtColType, EtaColType, RingerKerasTrainingJob


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type DatasetTypeType = Literal["train", "val", "test"]

type DatasetTypesType = Annotated[
    list[DatasetTypeType],
    Field(
        description=("Dataset splits to evaluate.  Each entry must be one of 'train', 'val', or 'test'."),
        min_length=1,
    ),
]


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------


class ReferencePoint(BaseModel):
    """A named operating point defined by a target true-positive rate.

    Each reference point describes a desired working point on the ROC curve.
    During the threshold-fit job the decision threshold whose empirical TPR is
    closest (in relative error) to :attr:`tpr` is selected and recorded.

    Attributes
    ----------
    tpr : float
        Target true-positive rate, in ``(0, 1]``.
    color : str
        Matplotlib colour string used to draw this point on ROC plots.
    label : str
        Human-readable name shown in plot legends and summary reports.

    Examples
    --------
    >>> ref = ReferencePoint(tpr=0.995, color="red", label="Tight")
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tpr: Annotated[
        float,
        Field(
            description="Target true positive rate for this operating point.",
            gt=0.0,
            le=1.0,
        ),
    ]
    color: Annotated[
        str,
        Field(description="Matplotlib colour string used when plotting this reference."),
    ]
    label: Annotated[
        str,
        Field(description="Human-readable label shown in plots and reports."),
    ]


# ---------------------------------------------------------------------------
# Main job class
# ---------------------------------------------------------------------------


class RingerCommitteeThresholdFitJob(YamlBaseModel):
    """Fit decision thresholds for a trained Ringer specialist committee.

    This job evaluates a previously trained committee (produced by
    :class:`~neuralnet.workflows.ringer.training.RingerKerasTrainingJob`) on
    a configurable dataset and, for each specialist model and each named
    reference operating point, finds the decision threshold whose empirical
    TPR is closest to the target.

    After :meth:`submit` completes the ``output_path`` directory holds all
    evaluation outputs, including:
    - ``models/member_{id}/`` containing the member model (``model.keras``) and
      confusion matrix parquet files for each dataset split.
    - ``{op_point}.json`` (e.g. ``tight.json``, ``medium.json``) containing all
      pipeline configs and decision thresholds for each operating point.
    - ``results.parquet`` with unnested member-level metrics.
    - ``committee_results.json`` with committee-wide aggregate metrics.

    The fitted committee pipeline can later be reconstructed from scratch via
    :meth:`get_specialist_committee` (or :meth:`get_committee`).

    Parameters
    ----------
    fit_job_path : Path
        Path to a completed ``RingerKerasTrainingJob`` output directory.
    dataset_dir : Path
        Root directory of the evaluation dataset (may differ from the
        training dataset).
    output_path : Path
        Directory where all evaluation outputs are written.
    references : dict[str, ReferencePoint]
        Named operating points to fit.  Keys become the reference names used
        throughout the output files (e.g. ``"tight"``, ``"medium"``).
    dataset_types : list[str], optional
        Which dataset splits to evaluate.  Defaults to
        ``["train", "val", "test"]``.
    batch_size : int, optional
        Inference batch size.  Defaults to ``1024``.
    score_col : str, optional
        Name of the model-output column produced by each specialist.
        Defaults to ``"output"``.
    logger_name : str or None, optional
        Logger name forwarded to :func:`logging.getLogger`.
    data_table, kfold_table, label_col, fold_col, rings_col, et_col, eta_col : str or None
        Optional overrides for dataset-schema column / table names.  When
        ``None`` (the default) the corresponding value is inherited from the
        loaded training job.

    Examples
    --------
    Minimal YAML configuration (all dataset fields inherited from training)::

        fit_job_path: /data/jobs/training_output
        dataset_dir:  /data/datasets/evaluation
        output_path:  /data/jobs/threshold_fit
        references:
          tight:
            tpr:   0.995
            color: red
            label: Tight
          medium:
            tpr:   0.996
            color: orange
            label: Medium

    Loading the fitted committee for deployment::

        committee = RingerCommitteeThresholdFitJob.get_specialist_committee(
            path="/data/jobs/threshold_fit",
            op_point="tight",
            dataset_type="test",
        )
        predictions = committee.predict(data_df, passthrough=True)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ------------------------------------------------------------------
    # Required fields
    # ------------------------------------------------------------------

    fit_job_path: Annotated[
        Path,
        Field(
            description=(
                "Path to the output directory of a completed "
                "RingerKerasTrainingJob.  Used to load the trained models and "
                "their metadata."
            ),
        ),
        PlainSerializer(str, return_type=str),
    ]

    dataset_dir: DirectoryType

    output_path: Annotated[
        Path,
        Field(description="Directory where evaluation outputs are written."),
        PlainSerializer(str, return_type=str),
    ]

    references: Annotated[
        dict[str, ReferencePoint],
        Field(
            description=(
                "Mapping from reference name (e.g. 'tight', 'medium') to a "
                "ReferencePoint that specifies the target TPR and plot styling."
            ),
            min_length=1,
        ),
    ]

    # ------------------------------------------------------------------
    # Optional / defaulted fields
    # ------------------------------------------------------------------

    dataset_types: DatasetTypesType = Field(
        default_factory=lambda: ["train", "val", "test"],
    )

    batch_size: Annotated[
        int,
        Field(
            gt=0,
            description="Batch size used for specialist-model inference.",
        ),
    ] = 1024

    score_col: Annotated[
        str,
        Field(description="Name of the model output (score) column."),
    ] = "output"

    logger_name: LoggerName = None

    # ------------------------------------------------------------------
    # Optional dataset-schema overrides
    #
    # Each field defaults to None, which means the value is inherited from
    # the loaded RingerKerasTrainingJob.  Supply a non-None value to run
    # evaluation on a dataset whose schema differs from the training one.
    # ------------------------------------------------------------------

    data_table: Annotated[
        DataTableType | None,
        Field(
            default=None,
            description=(
                "Name of the data table in the evaluation dataset.  Falls back to the training job's value when None."
            ),
        ),
    ] = None

    kfold_table: Annotated[
        KFoldTableType | None,
        Field(
            default=None,
            description=(
                "Name of the k-fold table in the evaluation dataset.  Falls back to the training job's value when None."
            ),
        ),
    ] = None

    label_col: Annotated[
        LabelColType | None,
        Field(
            default=None,
            description=("Name of the label column.  Falls back to the training job's value when None."),
        ),
    ] = None

    fold_col: Annotated[
        FoldColType | None,
        Field(
            default=None,
            description=("Name of the fold column.  Falls back to the training job's value when None."),
        ),
    ] = None

    rings_col: Annotated[
        RingsColType | None,
        Field(
            default=None,
            description=("Name of the rings column.  Falls back to the training job's value when None."),
        ),
    ] = None

    et_col: Annotated[
        EtColType | None,
        Field(
            default=None,
            description=("Name of the Et column.  Falls back to the training job's value when None."),
        ),
    ] = None

    eta_col: Annotated[
        EtaColType | None,
        Field(
            default=None,
            description=("Name of the Eta column.  Falls back to the training job's value when None."),
        ),
    ] = None

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------

    @cached_property
    def config_path(self) -> Path:
        """Path to the serialised job configuration JSON file."""
        return self.output_path / "config.json"

    @cached_property
    def models_dir(self) -> Path:
        """Path to the directory containing all member model directories."""
        return self.output_path / "models"

    def get_member_dir(self, member_id: int | str) -> Path:
        """Path to the output directory for a specific specialist member."""
        return self.models_dir / f"member_{member_id}"

    def get_member_model_path(self, member_id: int | str) -> Path:
        """Path to the Keras model file for a specific specialist member."""
        return self.get_member_dir(member_id) / "model.keras"

    def get_member_confusion_matrix_path(self, member_id: int | str, dataset_type: str) -> Path:
        """Path to the confusion matrix parquet file for a specific member and split."""
        return self.get_member_dir(member_id) / f"{dataset_type}_confusion_matrix.parquet"

    @cached_property
    def results_path(self) -> Path:
        """Path to the per-member results parquet file.

        One row per specialist member.  Columns are a flat dot-separated
        expansion of each member's training metadata merged with threshold-fit
        statistics for every ``(dataset_type, reference)`` combination.
        """
        return self.output_path / "results.parquet"

    @cached_property
    def committee_results_path(self) -> Path:
        """Path to the committee-level aggregate results JSON file.

        Contains summed confusion-matrix counts and derived rates (tpr, fpr,
        accuracy, sp) across all specialists for every
        ``(dataset_type, reference)`` combination.
        """
        return self.output_path / "committee_results.json"

    def get_op_point_json_path(self, op_point: str) -> Path:
        """Path to the JSON configuration file for a specific operating point.

        Parameters
        ----------
        op_point : str
            Name of the reference operating point (e.g. ``"tight"``).

        Returns
        -------
        Path
            Path to ``{output_path}/{op_point}.json``.
        """
        return self.output_path / f"{op_point}.json"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """Run the full committee threshold-fit pipeline.

        Steps
        -----
        1. Persist the job configuration to ``config.json``.
        2. Load the training job and build the specialist committee.
        3. For each specialist:

           a. Create member directory at ``models/member_{id}/``.
           b. Copy or export specialist member model to ``models/member_{id}/model.keras``.
           c. Run inference on the evaluation dataset (``predict_df``).
           d. For each dataset split filter the predictions by the
              corresponding ``is_{dataset_type}`` boolean column.
           e. Compute the full confusion-matrix curve via
              :func:`~neuralnet.metrics.enhanced_confusion_matrix_from_polars`.
           f. Save the confusion matrix DataFrame to
              ``models/member_{id}/{dataset_type}_confusion_matrix.parquet``.
           g. For each reference operating point, locate the threshold whose
              empirical TPR is closest to the target TPR.

        4. Aggregate per-member results into ``results.parquet``.
        5. Sum confusion-matrix counts across specialists and save
           ``committee_results.json``.
        6. For each reference operating point, export ``{op_point}.json``
           containing all preprocessing configurations, bin definitions,
           thresholds, and relative paths to the model files in ``models/``.
        """
        logger = logging.getLogger(self.logger_name)
        logger.info(f"Starting threshold-fit job. fit_job_path={self.fit_job_path}, output_path={self.output_path}")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Persist configuration for reproducibility.
        self.config_path.write_text(self.model_dump_json(indent=4), encoding="utf-8")

        training_job = RingerKerasTrainingJob.load(self.fit_job_path)
        dataset_kwargs = self._resolve_dataset_kwargs(training_job)
        specialist_committee = training_job.get_specialist_committee(
            et_col=dataset_kwargs["et_col"],
            eta_col=dataset_kwargs["eta_col"],
            rings_col=dataset_kwargs["rings_col"],
        )
        logger.info(f"Loaded specialist committee with {len(specialist_committee.models)} members.")

        # The label_col to use when filtering / computing confusion matrices.
        resolved_label_col = dataset_kwargs["label_col"]

        all_training_results: list[dict] = []
        # committee_thresholds: member_id -> dataset_type -> ref_name -> {threshold, tpr, fpr}
        committee_thresholds: dict[str, dict] = {}

        for i, specialist in enumerate(specialist_committee.models):
            et_bin = specialist.training_results["et_bin"]
            eta_bin = specialist.training_results["eta_bin"]
            model_id = specialist.training_results["id"]
            fold = specialist.training_results["fold"]
            member_dir = self.get_member_dir(model_id)
            member_dir.mkdir(parents=True, exist_ok=True)

            # Export/copy specialist model to models/member_{id}/model.keras
            member_model_file = self.get_member_model_path(model_id)
            orig_model_path = training_job.get_member_model_path(model_id)
            if orig_model_path.exists():
                shutil.copy2(orig_model_path, member_model_file)
            else:
                specialist.keras_model.save(member_model_file)

            logger.info(f"{i} - Evaluating member {model_id}: et_bin={et_bin}, eta_bin={eta_bin}")

            # Build dataset merging training-job defaults with any overrides.
            dataset = training_job.get_dataset(
                fold=fold,
                et_bin=et_bin,
                eta_bin=eta_bin,
                **dataset_kwargs,
            )

            logger.info("Predicting on the evaluation dataset…")
            prediction_df = dataset.predict_df().pipe(
                specialist.predict,
                passthrough=True,
                batch_size=self.batch_size,
            )

            # Collect once; re-filter in Python for each split.
            prediction_collected = prediction_df.collect()

            member_results = specialist.training_results.copy()
            member_thresholds: dict[str, dict] = {}

            for dataset_type in self.dataset_types:
                logger.info(f"Computing confusion matrix for '{dataset_type}' split…")
                split_df = prediction_collected.filter(pl.col(f"is_{dataset_type}"))
                # # enhanced_confusion_matrix_from_polars groups by score_col
                # # (descending) and then drops it in its .select() step. We
                # # re-attach the sorted unique threshold values so that we can
                # # look up the fitted threshold value later.
                # score_thresholds = (
                #     split_df
                #     .select(pl.col(self.score_col))
                #     .drop_nulls()
                #     .unique()
                #     .sort(self.score_col, descending=True)
                #     .rename({self.score_col: "threshold"})
                # )
                cm_df = enhanced_confusion_matrix_from_polars(
                    data=split_df,
                    score_col=self.score_col,
                    label_col=resolved_label_col,
                )

                threshold_fit_results: dict[str, dict] = {}
                split_thresholds: dict[str, dict] = {}

                for reference_name, reference_info in self.references.items():
                    reference_tpr = reference_info.tpr
                    logger.info(
                        f"Fitting threshold to TPR={reference_tpr} "
                        f"for reference '{reference_name}' in '{dataset_type}' split…"
                    )
                    best_row = (
                        cm_df.with_columns(
                            (pl.col("tpr") - reference_tpr)
                            .abs()
                            .truediv(reference_tpr)
                            .mul(100)
                            .alias("relative_tpr_err")
                        )
                        .sort("relative_tpr_err")
                        .head(5)
                        .row(0, named=True)
                    )
                    threshold_fit_results[reference_name] = best_row
                    threshold_fit_results[reference_name]["ref"] = reference_info.model_dump()

                    # Record the threshold in the manifest.
                    split_thresholds[reference_name] = {
                        "threshold": best_row["threshold"],
                        "tpr": best_row["tpr"],
                        "fpr": best_row["fpr"],
                    }

                member_thresholds[dataset_type] = split_thresholds

                # Save confusion matrix DataFrame for this member and dataset split.
                cm_path = self.get_member_confusion_matrix_path(model_id, dataset_type)
                cm_df.write_parquet(cm_path)
                logger.info(f"Saved confusion matrix DataFrame to {cm_path}.")

                # Store threshold fit results (without the heavy cm DataFrame).
                member_results[dataset_type] = threshold_fit_results

            committee_thresholds[str(model_id)] = member_thresholds
            all_training_results.append(member_results)

        # ------------------------------------------------------------------
        # Aggregate per-member results.
        # ------------------------------------------------------------------
        logger.info("Aggregating per-member results…")
        all_training_results_df = pl.from_records(all_training_results)
        all_training_results_df = unnest_structs(all_training_results_df)
        all_training_results_df.write_parquet(self.results_path)
        logger.info(f"Wrote member results to {self.results_path}.")

        # ------------------------------------------------------------------
        # Committee-level aggregates.
        # ------------------------------------------------------------------
        committee_results = self._compute_committee_results(all_training_results_df)
        with self.committee_results_path.open("w", encoding="utf-8") as f:
            json.dump(committee_results, f, indent=4)
        logger.info(f"Wrote committee results to {self.committee_results_path}.")

        # ------------------------------------------------------------------
        # Export operating-point JSON files ({op_point}.json)
        # ------------------------------------------------------------------
        primary_split = "test" if "test" in self.dataset_types else self.dataset_types[0]

        for ref_name, ref_info in self.references.items():
            models_data = []
            for specialist in specialist_committee.models:
                model_id = specialist.training_results["id"]
                member_id_str = str(model_id)
                threshold_for_primary = committee_thresholds[member_id_str][primary_split][ref_name]["threshold"]

                # Serialize bins
                bins_data = []
                for b in specialist.bins:
                    b_type = "AbsoluteVariableBin" if isinstance(b, AbsoluteVariableBin) else "VariableBin"
                    bins_data.append(
                        {
                            "type": b_type,
                            "var_name": b.var_name,
                            "low": b.low,
                            "high": b.high,
                            "closed": b.closed,
                        }
                    )

                # Preprocessing pipeline config
                prep_data = {
                    "rings_col": dataset_kwargs["rings_col"],
                    "ring_fraction": training_job.ring_fraction,
                    "norm_strategy": training_job.norm_strategy,
                }

                # Threshold fit results across evaluated splits
                split_threshold_data = {}
                for dt in self.dataset_types:
                    split_threshold_data[dt] = committee_thresholds[member_id_str][dt][ref_name]

                models_data.append(
                    {
                        "id": model_id,
                        "fold": specialist.training_results.get("fold"),
                        "model_path": f"models/member_{model_id}/model.keras",
                        "decision_threshold": threshold_for_primary,
                        "bins": bins_data,
                        "preprocessing": prep_data,
                        "threshold_fit_results": split_threshold_data,
                        "training_results": specialist.training_results,
                    }
                )

            op_point_data = {
                "op_point": ref_name,
                "reference": ref_info.model_dump(),
                "preprocessing": {
                    "rings_col": dataset_kwargs["rings_col"],
                    "ring_fraction": training_job.ring_fraction,
                    "norm_strategy": training_job.norm_strategy,
                },
                "dataset_types": self.dataset_types,
                "models": models_data,
            }

            op_point_json_path = self.get_op_point_json_path(ref_name)
            with op_point_json_path.open("w", encoding="utf-8") as f:
                json.dump(cast_to_json_value(op_point_data), f, indent=4)
            logger.info(f"Wrote operating point configuration to {op_point_json_path}.")

    @classmethod
    def get_specialist_committee(
        cls,
        path: Path | str,
        op_point: str,
        dataset_type: DatasetTypeType = "test",
        et_col: str | None = None,
        eta_col: str | None = None,
        rings_col: str | None = None,
    ) -> BinnedSpecialistCommittee:
        """Load a specialist committee pipeline with fitted decision thresholds.

        Reconstructs the full :class:`~neuralnet.workflows.ringer.models.BinnedSpecialistCommittee`
        from an exported operating-point JSON configuration file (e.g. ``tight.json``).
        All bins, preprocessing pipelines, model weights, and decision thresholds are
        built directly from the threshold-fit directory without needing the original
        training job.

        Parameters
        ----------
        path : Path or str
            Directory produced by :meth:`submit` (or direct path to an
            operating-point JSON file).
        op_point : str
            Name of the reference operating point (e.g. ``"tight"``, ``"medium"``).
        dataset_type : {"train", "val", "test"}, optional
            Dataset split from which to read the fitted threshold. Defaults to ``"test"``.
        et_col : str or None, optional
            Optional override for the Et column name.
        eta_col : str or None, optional
            Optional override for the Eta column name.
        rings_col : str or None, optional
            Optional override for the Rings column name.

        Returns
        -------
        BinnedSpecialistCommittee
            Committee with one ``BinnedSpecialistModel`` per selected specialist,
            each pre-configured with the fitted ``decision_threshold``.

        Raises
        ------
        FileNotFoundError
            If ``path`` or the operating-point JSON file does not exist.
        """
        return BinnedSpecialistCommittee.from_json(
            path=path,
            op_point=op_point,
            dataset_type=dataset_type,
            et_col=et_col,
            eta_col=eta_col,
            rings_col=rings_col,
        )

    @classmethod
    def get_committee(
        cls,
        path: Path | str,
        op_point: str,
        dataset_type: DatasetTypeType = "test",
        et_col: str | None = None,
        eta_col: str | None = None,
        rings_col: str | None = None,
    ) -> BinnedSpecialistCommittee:
        """Alias for :meth:`get_specialist_committee`."""
        return cls.get_specialist_committee(
            path=path,
            op_point=op_point,
            dataset_type=dataset_type,
            et_col=et_col,
            eta_col=eta_col,
            rings_col=rings_col,
        )

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load a ``RingerCommitteeThresholdFitJob`` from a saved output directory.

        Parameters
        ----------
        path : Path or str
            Directory that was used as ``output_path`` during a previous
            :meth:`submit` call. Must contain a ``config.json`` file.

        Returns
        -------
        RingerCommitteeThresholdFitJob
            Reconstructed job instance.

        Raises
        ------
        FileNotFoundError
            If ``path`` or ``config.json`` do not exist.
        NotADirectoryError
            If ``path`` is not a directory.
        """
        path = Path(path)
        cls._validate_saved_directory(path)
        with (path / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(**config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_dataset_kwargs(self, training_job: RingerKerasTrainingJob) -> dict:
        """Merge training-job dataset defaults with evaluation-job overrides.

        For every overridable dataset field declared on this job, the
        user-supplied value takes precedence when it is not ``None``; otherwise
        the corresponding attribute from ``training_job`` is used.

        Parameters
        ----------
        training_job : RingerKerasTrainingJob
            The loaded training job whose configuration supplies the default
            column / table names.

        Returns
        -------
        dict
            Resolved keyword-argument mapping ready to be forwarded to
            :meth:`RingerKerasTrainingJob.get_dataset`. Always contains the
            following keys: ``dataset_dir``, ``data_table``, ``kfold_table``,
            ``label_col``, ``fold_col``, ``rings_col``, ``et_col``,
            ``eta_col``.
        """
        overridable: dict[str, str] = {
            "dataset_dir": "dataset_dir",
            "data_table": "data_table",
            "kfold_table": "kfold_table",
            "label_col": "label_col",
            "fold_col": "fold_col",
            "rings_col": "rings_col",
            "et_col": "et_col",
            "eta_col": "eta_col",
        }
        resolved: dict = {}
        for field, training_attr in overridable.items():
            override_value = getattr(self, field, None)
            if override_value is not None:
                resolved[field] = override_value
            else:
                resolved[field] = getattr(training_job, training_attr)
        return resolved

    def _compute_committee_results(
        self,
        all_training_results_df: pl.DataFrame,
    ) -> dict:
        """Compute committee-level aggregate metrics.

        Sums per-member confusion-matrix counts across all specialist models
        and derives aggregate ``tpr``, ``fpr``, ``accuracy``, and ``sp`` for
        every ``(dataset_type, reference)`` combination.

        Parameters
        ----------
        all_training_results_df : pl.DataFrame
            Flat (unnested) per-member results DataFrame, one row per member.
            Expected to contain columns named
            ``"{dataset_type}.{ref}.{metric}"`` for every ``(dataset_type,
            ref)`` pair in the job configuration.

        Returns
        -------
        dict
            Flat mapping of ``"{dataset_type}.{ref}.{metric}"`` keys to scalar
            values, compatible with JSON serialisation.
        """
        exprs: list[pl.Expr] = []
        second_level_exprs: list[pl.Expr] = []

        for dataset_type, ref_type in product(self.dataset_types, self.references):
            prefix = f"{dataset_type}.{ref_type}"
            tp = pl.col(f"{prefix}.tp").sum()
            tn = pl.col(f"{prefix}.tn").sum()
            fp = pl.col(f"{prefix}.fp").sum()
            fn = pl.col(f"{prefix}.fn").sum()
            correct = pl.col(f"{prefix}.correct").sum()
            incorrect = pl.col(f"{prefix}.incorrect").sum()
            positives = pl.col(f"{prefix}.positives").sum()
            negatives = pl.col(f"{prefix}.negatives").sum()
            total = pl.col(f"{prefix}.total").sum()

            exprs.extend([tp, tn, fp, fn, correct, incorrect, positives, negatives, total])

            tpr_expr = tp.truediv(positives)
            fpr_expr = fp.truediv(negatives)
            second_level_exprs.extend(
                [
                    tpr_expr.alias(f"{prefix}.tpr"),
                    fpr_expr.alias(f"{prefix}.fpr"),
                    correct.truediv(total).alias(f"{prefix}.accuracy"),
                    polars_sp_index(tpr_expr, fpr_expr).alias(f"{prefix}.sp"),
                ]
            )

        committee_df = all_training_results_df.select(*exprs).with_columns(*second_level_exprs)
        return committee_df.row(0, named=True)

    @staticmethod
    def _validate_saved_directory(path: Path) -> None:
        """Assert that ``path`` is a completed threshold-fit output directory.

        Parameters
        ----------
        path : Path
            Directory to validate.

        Raises
        ------
        FileNotFoundError
            If ``path`` or ``config.json`` does not exist.
        NotADirectoryError
            If ``path`` is not a directory.
        """
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        if not path.is_dir():
            raise NotADirectoryError(f"Path {path} is not a directory.")
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found in {path}. Has RingerCommitteeThresholdFitJob.submit() been called?"
            )
