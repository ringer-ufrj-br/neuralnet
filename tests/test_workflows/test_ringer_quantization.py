import os
import pytest
import numpy as np
import polars as pl

from neuralnet.bins import VariableBin
from neuralnet.quantization.quantizers import FixedPointQuantizer
from neuralnet.quantization.keras import FixedPointQuantizationDense
from neuralnet.workflows.ringer.models import BinnedSpecialistCommittee, BinnedSpecialistModel


RINGS_COL = "rings"
RING_FRACTION = 2
N_RINGS = 100  # total rings in the dummy data (> 8+64+8+8+4+4+4 = 100)
N_SAMPLES = 20


def make_dummy_data() -> pl.DataFrame:
    """Build a minimal Polars DataFrame with rings, et, eta and id columns."""
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "id": np.arange(N_SAMPLES, dtype=np.int64),
            RINGS_COL: [rng.random(N_RINGS, dtype=np.float32).tolist() for _ in range(N_SAMPLES)],
            "et": np.linspace(0.0, 99.0, N_SAMPLES, dtype=np.float32),
            "eta": np.linspace(0.0, 2.4, N_SAMPLES, dtype=np.float32),
        }
    )


def build_preprocessing_pipeline():
    """Build the actual PreprocessingPipeline used in training.py."""
    from neuralnet.workflows.ringer.training import PreprocessingPipeline

    return PreprocessingPipeline.from_job_params(
        rings_col=RINGS_COL,
        ring_fraction=RING_FRACTION,
        norm_strategy="l1",
    )


def build_test_binned_specialist_model():
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import keras

    preprocessing = build_preprocessing_pipeline()
    n_features = len(preprocessing.output_cols)

    keras_model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    keras_model(np.zeros((1, n_features), dtype=np.float32))

    bins = [VariableBin(var_name="et", low=0.0, high=100.0, closed="left")]

    return BinnedSpecialistModel(
        bins=bins,
        keras_model=keras_model,
        preprocessing=preprocessing,
        decision_threshold=0.5,
        fold=0,
        training_results={"id": 1, "score": 0.95},
    )


def test_binned_specialist_model_fixed_point_quantization():
    model = build_test_binned_specialist_model()
    norm_quantizer = FixedPointQuantizer(integer_bits=1, fractional_bits=7)
    nn_quantizer = FixedPointQuantizer(integer_bits=2, fractional_bits=6)

    quantized_model = model.fixed_point_quantization(
        norm_quantizer=norm_quantizer,
        nn_quantizer=nn_quantizer,
    )

    assert isinstance(quantized_model, BinnedSpecialistModel)
    assert quantized_model.fold == model.fold
    assert quantized_model.decision_threshold == model.decision_threshold
    assert quantized_model.training_results == model.training_results

    # Preprocessing should have been quantized to FixedPointQuantizedAlternativeNorm1
    from neuralnet.normalizers.polars import FixedPointQuantizedAlternativeNorm1
    assert isinstance(quantized_model.preprocessing.normalizer, FixedPointQuantizedAlternativeNorm1)

    # Dense layers should be converted to FixedPointQuantizationDense
    dense_layers = [l for l in quantized_model.keras_model.layers if isinstance(l, FixedPointQuantizationDense)]
    assert len(dense_layers) == 2


def test_binned_specialist_model_quantization_predict_runs():
    """Verify the quantized model can run inference on real data."""
    model = build_test_binned_specialist_model()
    norm_quantizer = FixedPointQuantizer(integer_bits=1, fractional_bits=7)
    nn_quantizer = FixedPointQuantizer(integer_bits=2, fractional_bits=6)

    quantized_model = model.fixed_point_quantization(
        norm_quantizer=norm_quantizer,
        nn_quantizer=nn_quantizer,
    )

    data = make_dummy_data()
    result = quantized_model.predict(data, passthrough=False)
    assert isinstance(result, pl.DataFrame)
    assert "output" in result.columns
    assert "prediction" in result.columns
    assert result.height == N_SAMPLES


def test_binned_specialist_committee_fixed_point_quantization():
    m1 = build_test_binned_specialist_model()
    m2 = build_test_binned_specialist_model()
    committee = BinnedSpecialistCommittee(models=[m1, m2])

    norm_quantizer = FixedPointQuantizer(integer_bits=1, fractional_bits=7)
    nn_quantizer = FixedPointQuantizer(integer_bits=2, fractional_bits=6)

    quantized_committee = committee.fixed_point_quantization(
        norm_quantizer=norm_quantizer,
        nn_quantizer=nn_quantizer,
    )

    assert isinstance(quantized_committee, BinnedSpecialistCommittee)
    assert len(quantized_committee.models) == 2

    from neuralnet.normalizers.polars import FixedPointQuantizedAlternativeNorm1
    for m in quantized_committee.models:
        dense_layers = [l for l in m.keras_model.layers if isinstance(l, FixedPointQuantizationDense)]
        assert len(dense_layers) == 2
        assert isinstance(m.preprocessing.normalizer, FixedPointQuantizedAlternativeNorm1)
