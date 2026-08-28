"""Ringer committee inference job.

This module provides :class:`RingerCommitteeInferenceJob`, a Pydantic-backed
workflow class that runs full-dataset inference using a fitted Ringer specialist
committee and streams the results directly to a parquet table keyed by ``id``.

Overview
--------
The job loads a completed
:class:`~neuralnet.workflows.ringer.threshold_fit.RingerCommitteeThresholdFitJob`
output directory, loads all fitted specialist committees for all configured
operating points (e.g. ``tight``, ``medium``, etc.), and runs inference on the
full ``predict_df()`` split of the supplied dataset.

Results are written directly to a parquet file using Polars streaming mode
(:meth:`polars.LazyFrame.sink_parquet`) without materialising the entire
prediction table in RAM, making it suitable for memory-constrained environments.
Output columns contain only the event ``id`` (for joining with other tables) and
prefixed inference scores and boolean decisions for every evaluated operating
point (e.g. ``tight.output``, ``tight.prediction``, ``medium.output``,
``medium.prediction``).

Output directory layout
-----------------------
After :meth:`RingerCommitteeInferenceJob.submit` completes, the
``output_path`` directory contains the following files::

    output_path/
    ├── config.json
    │   Job configuration dumped as JSON for reproducibility.
    │
    └── inference_results.parquet
        Inference output table.  Columns:
          • id                (UInt64) — event identifier for joining with other tables.
          • {op}.output       (Float32) — committee output score for operating point {op}.
          • {op}.prediction   (Boolean) — thresholded decision for operating point {op}.

"""

import json
import logging
from functools import cached_property
from pathlib import Path
from typing import Annotated
import polars as pl
from pydantic import ConfigDict, Field, PlainSerializer

from ...datasets import DirectoryType
from ...datasets.ringer import (
    DataTableType,
    FoldColType,
    KFoldTableType,
    LabelColType,
    RingsColType,
    RingerParquetDataset,
)
from ...logging import LoggerName
from ...pydantic import YamlBaseModel
from .threshold_fit import RingerCommitteeThresholdFitJob
from .training import EtColType, EtaColType, RingerKerasTrainingJob


class RingerCommitteeInferenceJob(YamlBaseModel):
    """Run full-dataset inference with a fitted Ringer specialist committee.

    This job loads a specialist committee previously trained and threshold-fitted
    by :class:`~neuralnet.workflows.ringer.threshold_fit.RingerCommitteeThresholdFitJob`,
    runs inference on the full ``predict_df()`` split of the supplied dataset for
    all operating points, and streams the predictions to a parquet table using Polars
    streaming execution (:meth:`polars.LazyFrame.sink_parquet`).

    The resulting parquet table contains only the ``id`` column and the prefixed
    predictions for each operating point (e.g. ``tight.output``, ``tight.prediction``,
    ``medium.output``, ``medium.prediction``).

    Parameters
    ----------
    job_path : Path
        Path to the output directory of a completed
        ``RingerCommitteeThresholdFitJob``. Must contain the operating-point
        JSON files (e.g. ``tight.json``, ``medium.json``) produced by that job.
    dataset_dir : Path
        Root directory of the dataset to run inference on. May differ from
        the dataset used during training or threshold fitting.
    output_path : Path
        Directory where all inference outputs are written.
    op_points : list[str] or None, optional
        List of operating point names to evaluate (e.g. ``["tight", "medium"]``).
        When ``None`` (the default), all operating points found in ``job_path``
        are evaluated.
    batch_size : int, optional
        Inference batch size. Defaults to ``1024``.
    logger_name : str or None, optional
        Logger name forwarded to :func:`logging.getLogger`.
    data_table, kfold_table, label_col, fold_col, rings_col, et_col, eta_col : str or None
        Optional overrides for dataset-schema column / table names. When
        ``None`` (the default), the corresponding value is inherited from the
        training job that was used to produce the threshold-fit directory.

    Examples
    --------
    Minimal YAML configuration::

        job_path:     /data/jobs/threshold_fit
        dataset_dir:  /data/datasets/inference
        output_path:  /data/jobs/inference

    Loading the inference results afterwards::

        import polars as pl
        df = pl.read_parquet("/data/jobs/inference/inference_results.parquet")
        # df has columns: id, tight.output, tight.prediction, medium.output, ...
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ------------------------------------------------------------------
    # Required fields
    # ------------------------------------------------------------------

    job_path: Annotated[
        Path,
        Field(
            description=(
                "Path to the output directory of a completed "
                "RingerCommitteeThresholdFitJob. Must contain the operating-point "
                "JSON files (e.g. tight.json, medium.json) produced by that job."
            ),
        ),
        PlainSerializer(str, return_type=str),
    ]

    dataset_dir: DirectoryType

    output_path: Annotated[
        Path,
        Field(description="Directory where inference outputs are written."),
        PlainSerializer(str, return_type=str),
    ]

    max_rows_per_file: Annotated[
        int,
        Field(
            gt=0,
            description=(
                "Maximum number of rows per output parquet file. "
                "Defaults to 1_000_000. Set to None to disable splitting."
            ),
        ),
    ] = 1_000_000

    # ------------------------------------------------------------------
    # Optional / defaulted fields
    # ------------------------------------------------------------------

    op_points: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "List of operating point names to evaluate (e.g. ['tight', 'medium']). "
                "If None (the default), evaluates all operating points found in the threshold fit job."
            ),
        ),
    ] = None

    batch_size: Annotated[
        int,
        Field(
            gt=0,
            description="Batch size used for committee model inference.",
        ),
    ] = 1024

    logger_name: LoggerName = None

    # ------------------------------------------------------------------
    # Optional dataset-schema overrides
    #
    # Each field defaults to None, which means the value is inherited from
    # the training job that was used to produce the threshold-fit directory.
    # Supply a non-None value to run inference on a dataset whose schema
    # differs from the training one.
    # ------------------------------------------------------------------

    data_table: Annotated[
        DataTableType | None,
        Field(
            default=None,
            description=(
                "Name of the data table in the inference dataset. Falls back to the training job's value when None."
            ),
        ),
    ] = None

    kfold_table: Annotated[
        KFoldTableType | None,
        Field(
            default=None,
            description=(
                "Name of the k-fold table in the inference dataset. Falls back to the training job's value when None."
            ),
        ),
    ] = None

    label_col: Annotated[
        LabelColType | None,
        Field(
            default=None,
            description="Name of the label column. Falls back to the training job's value when None.",
        ),
    ] = None

    fold_col: Annotated[
        FoldColType | None,
        Field(
            default=None,
            description="Name of the fold column. Falls back to the training job's value when None.",
        ),
    ] = None

    rings_col: Annotated[
        RingsColType | None,
        Field(
            default=None,
            description="Name of the rings column. Falls back to the training job's value when None.",
        ),
    ] = None

    et_col: Annotated[
        EtColType | None,
        Field(
            default=None,
            description="Name of the Et column. Falls back to the training job's value when None.",
        ),
    ] = None

    eta_col: Annotated[
        EtaColType | None,
        Field(
            default=None,
            description="Name of the Eta column. Falls back to the training job's value when None.",
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
    def inference_results_path(self) -> Path:
        """Path to the inference results parquet file.

        Contains one row per event in the ``predict_df()`` split. Columns
        are ``id`` (UInt64) plus ``{op}.output`` (Float32) and
        ``{op}.prediction`` (Boolean) for each operating point.
        """
        return self.output_path / "inference_results.parquet"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """Run the full committee inference pipeline in streaming mode.

        Steps
        -----
        1. Persist the job configuration to ``config.json``.
        2. Load the threshold-fit job from ``job_path`` and resolve dataset
           column / table name overrides against the training job.
        3. Discover operating points to evaluate and load their specialist
           committees.
        4. Build a :class:`~neuralnet.datasets.ringer.RingerParquetDataset`
           for the inference ``dataset_dir``.
        5. For each specialist member, filter the lazy dataset to its bin, run
           feature preprocessing and model inference once, and construct
           prefixed output and prediction columns for all operating points.
        6. Combine the specialist predictions as a :class:`polars.LazyFrame` and
           stream directly to disk with :meth:`polars.LazyFrame.sink_parquet`.
        """
        logger = logging.getLogger(self.logger_name)
        logger.info(f"Starting inference job. job_path={self.job_path}, output_path={self.output_path}")
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Persist configuration for reproducibility.
        self.config_path.write_text(self.model_dump_json(indent=4), encoding="utf-8")

        # ------------------------------------------------------------------
        # Step 1: Load threshold-fit job and resolve column/table overrides.
        # ------------------------------------------------------------------
        threshold_fit_job = RingerCommitteeThresholdFitJob.load(self.job_path)
        training_job = RingerKerasTrainingJob.load(threshold_fit_job.fit_job_path)
        dataset_kwargs = self._resolve_dataset_kwargs(training_job)

        logger.info(f"Loaded threshold-fit job from {self.job_path}. Resolved dataset_kwargs={dataset_kwargs}")

        # ------------------------------------------------------------------
        # Step 2: Determine operating points and load specialist committees.
        # ------------------------------------------------------------------
        if self.op_points is not None:
            op_points = list(self.op_points)
        elif hasattr(threshold_fit_job, "references") and threshold_fit_job.references:
            op_points = list(threshold_fit_job.references.keys())
        else:
            op_points = [
                p.stem for p in self.job_path.glob("*.json") if p.name not in ("config.json", "committee_results.json")
            ]

        if not op_points:
            raise ValueError(f"No operating point configurations found in {self.job_path}")

        logger.info(f"Evaluating operating points: {op_points}")

        committees = {
            op_point: RingerCommitteeThresholdFitJob.get_specialist_committee(
                path=self.job_path,
                op_point=op_point,
                et_col=dataset_kwargs["et_col"],
                eta_col=dataset_kwargs["eta_col"],
                rings_col=dataset_kwargs["rings_col"],
            )
            for op_point in op_points
        }

        # ------------------------------------------------------------------
        # Step 3: Build prediction lazy frame.
        # ------------------------------------------------------------------
        dataset = RingerParquetDataset(
            dataset_dir=dataset_kwargs["dataset_dir"],
            data_table=dataset_kwargs["data_table"],
            rings_col=dataset_kwargs["rings_col"],
            kfold_table=dataset_kwargs["kfold_table"],
            label_col=dataset_kwargs["label_col"],
            fold_col=dataset_kwargs["fold_col"],
            fold=0,
        )

        predicted_df = dataset.predict_df()
        selection_cols = {'id'}
        for op_point, committee in committees.items():
            logger.info(
                f"Running inference for operating point '{op_point}' with {len(committee.models)} specialist members."
            )
            rename_map = {col: f"{op_point}.{col}" for col in committee.output_cols if col != "id"}
            selection_cols.update(rename_map.values())
            predicted_df = (
                predicted_df
                .pipe(committee.predict, batch_size=self.batch_size, passthrough=True)
                .rename(rename_map)
            )
        predicted_df = predicted_df.select(*sorted(selection_cols))

        logger.info(f"Streaming inference results to {self.inference_results_path} using sink_parquet...")
        if self.max_rows_per_file is not None:
            predicted_df.sink_parquet(
                pl.PartitionBy(
                    str(self.inference_results_path),
                    max_rows_per_file=self.max_rows_per_file,
                ),
                compression="snappy",
            )
        else:
            predicted_df.sink_parquet(
                self.inference_results_path,
                compression="snappy",
            )

        logger.info("Completed streaming inference results.")

    @classmethod
    def load(cls, path: Path | str) -> "RingerCommitteeInferenceJob":
        """Load a ``RingerCommitteeInferenceJob`` from a saved output directory.

        Parameters
        ----------
        path : Path or str
            Directory that was used as ``output_path`` during a previous
            :meth:`submit` call. Must contain a ``config.json`` file.

        Returns
        -------
        RingerCommitteeInferenceJob
            Reconstructed job instance.

        Raises
        ------
        FileNotFoundError
            If ``path`` or ``config.json`` do not exist.
        NotADirectoryError
            If ``path`` is not a directory.
        """
        path = Path(path)
        _validate_saved_directory(path)
        with (path / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(**config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_dataset_kwargs(self, training_job: RingerKerasTrainingJob) -> dict:
        """Merge training-job dataset defaults with inference-job overrides.

        For every overridable dataset field declared on this job, the
        user-supplied value takes precedence when it is not ``None``; otherwise
        the corresponding attribute from ``training_job`` is used.

        Parameters
        ----------
        training_job : RingerKerasTrainingJob
            The training job whose configuration supplies the default column /
            table names.

        Returns
        -------
        dict
            Resolved keyword-argument mapping ready to be forwarded to
            :class:`~neuralnet.datasets.ringer.RingerParquetDataset`. Always
            contains: ``dataset_dir``, ``data_table``, ``kfold_table``,
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
        # Always use the inference dataset_dir, not the training one.
        resolved["dataset_dir"] = self.dataset_dir
        return resolved


def _validate_saved_directory(path: Path) -> None:
    """Assert that ``path`` is a completed inference output directory.

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
            f"config.json not found in {path}. Has RingerCommitteeInferenceJob.submit() been called?"
        )
