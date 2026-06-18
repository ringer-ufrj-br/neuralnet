def test_get_ring_slices_per_layer():
    from neuralnet.utils import get_ring_slices_per_layer
    slices_2 = get_ring_slices_per_layer(2)
    assert len(slices_2) == 50  # 100 total rings / 2
    
    slices_1 = get_ring_slices_per_layer(1)
    assert len(slices_1) == 100