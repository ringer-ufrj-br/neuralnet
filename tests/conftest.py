from pathlib import Path
from typing import Generator
import pytest
from concurrent.futures import ProcessPoolExecutor


@pytest.fixture
def root_dir() -> Path:
    return Path(__file__).parent.parent.absolute()


@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parent.absolute() / "data"


@pytest.fixture(scope="session")
def create_logs_dir() -> Path:
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True, parents=True)
    return logs_dir


@pytest.fixture()
def isolated_executor() -> Generator[ProcessPoolExecutor, None, None]:
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
        yield executor


# @pytest.fixture
# def keras_with_backend(monkeypatch: MonkeyPatch):
#     def _reload(backend: str):
#         # Dynamically set the environment variable for this test only
#         monkeypatch.setenv("KERAS_BACKEND", backend)

#         # Target the specific module evaluating the env var at top-level
#         import keras

#         # Force Python to re-execute the module with the new environment
#         return reload(keras)

#     yield _reload

#     # Teardown: Optionally restore defaults so later tests start clean
#     import keras

#     reload(keras)
