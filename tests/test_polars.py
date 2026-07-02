import pytest
import polars as pl

from neuralnet.polars import (
    AlternativeNorm1,
    FixedPointQuantizedAlternativeNorm1,
)
from neuralnet.quantization.quantizers import FixedPointQuantizer


def assert_nested_lists_close(actual: list[list[float]], expected: list[list[float]]) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == pytest.approx(expected_row)


def test_alternative_norm1_normalizes_list_column_with_default_output_name():
    data = pl.DataFrame(
        {
            "rings": [
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
            ]
        }
    )

    transform = AlternativeNorm1(input_col="rings", output_col_base="rings_alternative_norm1")

    result = transform(data)

    assert transform.output_col == "rings_alternative_norm1"
    assert result.columns == ["rings", "rings_alternative_norm1"]
    assert_nested_lists_close(
        result.get_column(transform.output_col).to_list(),
        [[1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0], [0.0, 0.0, 0.0]],
    )


def test_alternative_norm1_fixed_point_quantization_wraps_configuration():
    quantizer = FixedPointQuantizer(integer_bits=1, fractional_bits=1)

    transform = AlternativeNorm1(input_col="rings", output_col="rings_norm")
    quantized_transform = transform.fixed_point_quantization(quantizer)

    assert isinstance(quantized_transform, FixedPointQuantizedAlternativeNorm1)
    assert quantized_transform.input_col.meta.output_name() == "rings"
    assert quantized_transform.output_col == "rings_norm"
    assert quantized_transform.quantizer == quantizer


@pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
            [[0.25, 0.25, 0.5], [0.0, 0.0, 0.0]],
            # If there was no quantization, should have been [[0.16666667, 0.33333333, 0.5], [0, 0, 0]]
        ),
        (
            [[4.0, 5.0], [1.0, -1.0]],
            [[0.5, 0.5], [1.0, -1.0]],
            # If there was no quantization, should have been [[0.4444444, 0.5555555555], [1, -1]]
            # Alternative L1 does abs(sum(vector)) instead of the usual sum(abs(vector)), so the second row is not normalized to [0.5, -0.5] but to [1, -1]
        ),
    ],
)
@pytest.mark.parametrize("dtype", ['list', 'array'])
def test_fixed_point_quantized_alternative_norm1_quantizes_dataframe_and_lazyframe(
    frame_factory, input_data, expected_output, dtype
):
    quantizer = FixedPointQuantizer(integer_bits=1, fractional_bits=2)
    transform = FixedPointQuantizedAlternativeNorm1(
        input_col=["rings"],
        output_col_base="rings_quantized",
        quantizer=quantizer,
    )
    if dtype == 'array':
        dtype = pl.Array(pl.Float64, len(input_data[0]))
    elif dtype == 'list':
        dtype = pl.List(pl.Float64)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    data = frame_factory(
        {
            "rings": input_data
        },
        schema = {
            'rings': dtype
        }
    )

    result = transform(data)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()

    assert_nested_lists_close(
        result.get_column("rings_quantized").to_list(),
        expected_output,
    )
