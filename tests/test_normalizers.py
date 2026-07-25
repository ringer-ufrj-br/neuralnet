import pytest
import polars as pl
from polars.testing import assert_frame_equal

from neuralnet.normalizers.polars import (
    AlternativeNorm1,
    FixedPointQuantizedAlternativeNorm1,
)
from neuralnet.quantization.quantizers import FixedPointQuantizer


def assert_nested_lists_close(
    actual: list[list[float]], expected: list[list[float]]
) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == pytest.approx(expected_row)


@pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
def test_alternative_norm1(frame_factory):
    data = frame_factory(
        {"a": [1.0, 0.0, None], "b": [2.0, 0.0, 1.0], "c": [3.0, 0.0, 2.0]}
    )
    expected_result = pl.DataFrame(
        {
            "a_alternative_norm1": [1.0 / 6.0, 0.0, None],
            "b_alternative_norm1": [2.0 / 6.0, 0.0, None],
            "c_alternative_norm1": [3.0 / 6.0, 0.0, None],
        }
    )

    transform = AlternativeNorm1(input_cols=["a", "b", "c"], suffix="alternative_norm1")

    result = transform(data)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()

    assert transform.output_cols == [
        "a_alternative_norm1",
        "b_alternative_norm1",
        "c_alternative_norm1",
    ]
    assert_frame_equal(result, expected_result, check_dtype=True)


QUANTIZED_ALTERNATIVE_NORM1_SUFFIX = " alternative_norm1_quantized"


QUANTIZED_REFERENCE_DATA = {
    "standard_case": {
        "input_data": {"a": [1.0, 0.0], "b": [2.0, 0.0], "c": [3.0, 0.0]},
        "expected_output": {
            f"a_{QUANTIZED_ALTERNATIVE_NORM1_SUFFIX}": [0.25, 0.0],
            f"b_{QUANTIZED_ALTERNATIVE_NORM1_SUFFIX}": [0.25, 0.0],
            f"c_{QUANTIZED_ALTERNATIVE_NORM1_SUFFIX}": [0.5, 0.0],
        },
    },
    "norm_is_zero_case": {
        "input_data": {"a": [4.0, 1.0], "b": [5.0, -1.0]},
        "expected_output": {
            f"a_{QUANTIZED_ALTERNATIVE_NORM1_SUFFIX}": [0.5, 1.0],
            f"b_{QUANTIZED_ALTERNATIVE_NORM1_SUFFIX}": [0.5, -1.0],
        },
    },
    "null_case": {
        "input_data": {"a": [1.0, 0.0], "b": [2.0, 0.0], "c": [3.0, None]},
        "expected_output": {
            f"a_{QUANTIZED_ALTERNATIVE_NORM1_SUFFIX}": [0.25, None],
            f"b_{QUANTIZED_ALTERNATIVE_NORM1_SUFFIX}": [0.25, None],
            f"c_{QUANTIZED_ALTERNATIVE_NORM1_SUFFIX}": [0.5, None],
        },
    },
}


@pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (case_data["input_data"], case_data["expected_output"])
        for case_data in QUANTIZED_REFERENCE_DATA.values()
    ],
    ids=list(QUANTIZED_REFERENCE_DATA.keys()),
)
def test_fixed_point_quantized_alternative_norm1_quantizes_dataframe_and_lazyframe(
    frame_factory, input_data, expected_output
):
    quantizer = FixedPointQuantizer(integer_bits=1, fractional_bits=2)
    data = frame_factory(input_data)
    transform = FixedPointQuantizedAlternativeNorm1(
        input_cols=data.columns,
        suffix=QUANTIZED_ALTERNATIVE_NORM1_SUFFIX,
        quantizer=quantizer,
    )
    expected_result = pl.DataFrame(expected_output)

    result = transform(data)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()

    assert_frame_equal(result, expected_result, check_dtype=True)
