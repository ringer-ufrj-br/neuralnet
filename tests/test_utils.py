import pytest


def test_get_ring_slices_per_layer():
    from neuralnet.utils import get_ring_slices_per_layer
    slices_2 = get_ring_slices_per_layer(2)
    assert len(slices_2) == 50  # 100 total rings / 2
    
    slices_1 = get_ring_slices_per_layer(1)
    assert len(slices_1) == 100


def test_unflatten_mapping_builds_nested_dict():
    from neuralnet.utils import unflatten_mapping

    result = unflatten_mapping({"model.layer.units": 64, "model.layer.activation": "relu", "train.epochs": 10})

    assert result == {"model": {"layer": {"units": 64, "activation": "relu"}}, "train": {"epochs": 10}}


def test_unflatten_mapping_raises_on_conflicting_keys():
    from neuralnet.utils import unflatten_mapping

    with pytest.raises(KeyError, match="Duplicate key found"):
        unflatten_mapping({"a": 1, "a.b": 2})