import pytest
from pathlib import Path
import numpy as np
import torch
import polars as pl

from neuralnet.datasets.ringer import get_ring_slices_per_layer, Bin, RingerParquetDataset

def test_get_ring_slices_per_layer():
    slices_2 = get_ring_slices_per_layer(2)
    assert len(slices_2) == 50  # 100 total rings / 2
    
    slices_1 = get_ring_slices_per_layer(1)
    assert len(slices_1) == 100

@pytest.fixture
def ringer_dataset(test_data_dir: Path) -> RingerParquetDataset:
    dataset_dir = str(test_data_dir / "test_dataset")
    return RingerParquetDataset(
        dataset_dir=dataset_dir,
        data_table='electron_ringer.parquet',
        rings_col='TrigEMClusterContainer.ringsE',
        kfold_table='standard_binning_kfold.parquet',
        label_col='label',
        fold_col='kfold',
        et_col='TrigEMClusterContainer.et',
        eta_col='TrigEMClusterContainer.eta',
        et_bin=Bin(low=15000.0, high=20000.0),
        eta_bin=Bin(low=0.0, high=0.8),
        ring_fraction=2,
        batch_size=32
    )

def test_dataset_folds(ringer_dataset: RingerParquetDataset):
    assert ringer_dataset.get_n_folds() == 5
    assert ringer_dataset.current_fold == 0
    
    ringer_dataset.set_fold(1)
    assert ringer_dataset.current_fold == 1
    
    with pytest.raises(ValueError):
        ringer_dataset.set_fold(-1)
        
    with pytest.raises(ValueError):
        ringer_dataset.set_fold(5)

@pytest.mark.parametrize("split", ["train", "val", "test", "predict"])
def test_dataset_numpy(ringer_dataset: RingerParquetDataset, split: str):
    method = getattr(ringer_dataset, f"{split}_numpy")
    X, y = method()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[1] == 50
    assert len(X) == len(y)

@pytest.mark.parametrize("split", ["train", "val", "test", "predict"])
def test_dataset_dataloader(ringer_dataset: RingerParquetDataset, split: str):
    method = getattr(ringer_dataset, f"{split}_dataloader")
    dl = method()
    batch = next(iter(dl))
    
    assert isinstance(batch, (list, tuple))
    assert len(batch) == 2
    
    dataset_tensor = batch[0]
    label_tensor = batch[1]
    
    assert isinstance(dataset_tensor, torch.Tensor)
    assert isinstance(label_tensor, torch.Tensor)
    assert dataset_tensor.shape[1] == 50
    assert dataset_tensor.shape[0] <= 32
    assert label_tensor.shape[0] == dataset_tensor.shape[0]

