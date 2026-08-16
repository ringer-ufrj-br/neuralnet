from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, Field, PlainSerializer
import submitit


class DebugExecutor:
    def __init__(self, folder: Path):
        self.folder = folder

    def submit(self, fn, *args, **kwargs):
        # Just run the function directly for debugging
        return fn(*args, **kwargs)

    def update_parameters(self, **kwargs):
        # No parameters to update in the debug executor
        return self

    def batch(self, *args, **kwargs):
        # Just return self for chaining
        return self

    def __enter__(self):
        # No resources to acquire in the debug executor
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # No resources to release in the debug executor
        return False


class ExecutorConfig(BaseModel):
    """
    Configuration for code executor. It provides a unified interface for different types of executors, such as Slurm, local parallel execution, and direct debug execution.
    """

    cpus_per_task: Annotated[
        int, Field(description="Number of CPUs to allocate per task")
    ]
    executor_type: Annotated[
        Literal["slurm", "debug", "local"],
        Field(
            description='Type of executor to use. "slurm" submits the jobs to a slurm cluster, "debug" runs the jobs directly in the current process and "local" runs the jobs in parallel in the local machine using submitit LocalExecutor.'
        ),
    ]
    logs_dir: Annotated[
        Path,
        Field(description="Directory to save the logs of the slurm jobs"),
        PlainSerializer(str, return_type=str)
    ]
    name: Annotated[str, Field(description="Name of the job")]
    slurm_array_parallelism: Annotated[
        int,
        Field(description="Max number of jobs to run in parallel in the slurm array"),
    ] = 1
    slurm_partition: Annotated[
        str | None, Field(description="Slurm partition to submit the jobs to")
    ] = None
    stderr_to_stdout: Annotated[
        bool,
        Field(description="Whether to redirect stderr to stdout in the slurm jobs"),
    ] = True
    timeout_min: Annotated[
        int, Field(description="Timeout in minutes for the slurm jobs")
    ] = 24 * 60  # 1 day

    def model_post_init(self, context):
        if self.executor_type == "slurm" and self.slurm_partition is None:
            raise ValueError(
                "slurm_partition must be specified when using slurm executor"
            )
        return super().model_post_init(context)

    def update_executor_params(self, executor: submitit.Executor) -> submitit.Executor:
        executor.update_parameters(
            cpus_per_task=self.cpus_per_task,
            name=self.name,
            slurm_array_parallelism=self.slurm_array_parallelism,
            slurm_partition=self.slurm_partition,
            stderr_to_stdout=self.stderr_to_stdout,
            timeout_min=self.timeout_min,
        )
        return executor

    def get_executor(self) -> submitit.Executor:
        match self.executor_type:
            case "slurm":
                executor = submitit.AutoExecutor(folder=str(self.logs_dir))
            case "debug":
                # Usual submitit executors don't work in debug mode, so we use a custom executor that just runs the function directly.
                executor = DebugExecutor(folder=str(self.logs_dir))
            case "local":
                executor = submitit.LocalExecutor(folder=str(self.logs_dir))
            case _:
                raise ValueError(f"Invalid executor type: {self.executor_type}")

        executor = self.update_executor_params(executor)
        return executor
