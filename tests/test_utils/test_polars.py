from neuralnet.utils.polars import unnest_structs
import pytest
import polars as pl
from neuralnet.utils.polars import RingSlicesPerLayer, unnest_structs
from polars.testing import assert_frame_equal


class TestRingSlicesPerLayer:
    """Tests for RingSlicesPerLayer class."""

    def test_initialization_with_defaults(self):
        """Test basic initialization with required fields."""
        config = RingSlicesPerLayer(
            fraction=2,
            rings_col="rings",
        )
        assert config.fraction == 2
        assert config.output_format == "expanded_columns"
        assert config.rings_col == "rings"

    def test_initialization_with_custom_output_format(self):
        """Test initialization with custom output format."""
        config = RingSlicesPerLayer(
            fraction=2,
            output_format="expanded_columns",
            rings_col="rings",
        )
        assert config.output_format == "expanded_columns"

    def test_initialization_rejects_extra_fields(self):
        """Test that extra fields are rejected due to model_config."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            RingSlicesPerLayer(
                fraction=2,
                rings_col="rings",
                extra_field="not_allowed",
            )

    def test_fraction_must_be_positive(self):
        """Test that fraction must be greater than 0."""
        with pytest.raises(ValueError, match="greater than 0"):
            RingSlicesPerLayer(
                fraction=0,
                rings_col="rings",
            )

        with pytest.raises(ValueError, match="greater than 0"):
            RingSlicesPerLayer(
                fraction=-1,
                rings_col="rings",
            )

    def test_input_cols_property(self):
        """Test input_cols property returns the rings_col."""
        config = RingSlicesPerLayer(
            fraction=2,
            rings_col="my_rings",
        )
        assert config.input_cols == ["my_rings"]

    def test_output_cols_property(self):
        """Test output_cols property returns expanded column names."""
        config = RingSlicesPerLayer(
            fraction=2,
            rings_col="rings",
        )
        output_cols = config.output_cols
        assert isinstance(output_cols, list)
        assert len(output_cols) == 50  # 100 total rings / 2
        # Check that all output columns follow the naming convention
        assert all("rings." in col for col in output_cols)

    def test_get_expanded_column_name(self):
        """Test column name generation."""
        config = RingSlicesPerLayer(
            fraction=2,
            rings_col="rings",
        )
        col_name = config.get_expanded_column_name(5)
        assert col_name == "rings.5"

    def test_get_list_polars_expr(self):
        """Test generating Polars expression for list columns."""
        config = RingSlicesPerLayer(
            fraction=2,
            rings_col="rings",
        )
        expr = config.get_list_polars_expr()
        assert isinstance(expr, list)
        assert len(expr) > 0

    def test_get_array_polars_expr(self):
        """Test generating Polars expression for array columns."""
        config = RingSlicesPerLayer(
            fraction=2,
            rings_col="rings",
        )
        expr = config.get_array_polars_expr()
        assert isinstance(expr, list)
        assert len(expr) > 0

    def test_call_with_missing_column_raises_error(self):
        """Test that missing input column raises ValueError."""
        config = RingSlicesPerLayer(
            fraction=2,
            rings_col="nonexistent",
        )
        df = pl.DataFrame(
            {
                "rings": [
                    list(range(100)),
                ]
            }
        )

        with pytest.raises(ValueError, match="not found in the data schema"):
            config(df)

    def test_call_with_wrong_datatype_raises_error(self):
        """Test that wrong column type raises TypeError."""
        config = RingSlicesPerLayer(
            fraction=2,
            rings_col="rings",
        )
        df = pl.DataFrame(
            {
                "rings": [1, 2, 3]  # Integer column instead of list
            }
        )

        with pytest.raises(TypeError, match="Expected list or array column"):
            config(df)

    def test_call_with_array_of_ints_raises_error(self):
        """Test that array of non-floats raises TypeError."""
        config = RingSlicesPerLayer(
            fraction=2,
            rings_col="rings",
        )
        df = pl.DataFrame(
            {
                "rings": [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            }
        ).select(pl.col("rings").cast(pl.Array(pl.Int64, 3)))

        with pytest.raises(TypeError, match="Expected array of floats"):
            config(df)

    def test_call_with_invalid_type_raises_error(self):
        """Test that invalid data type (not DataFrame or LazyFrame) raises TypeError."""
        config = RingSlicesPerLayer(
            fraction=2,
            rings_col="rings",
        )

        with pytest.raises(TypeError, match="Expected pl.DataFrame or pl.LazyFrame"):
            config([1, 2, 3])

    def test_fraction_affects_output_cols_count(self):
        """Test that different fractions produce different output column counts."""
        config_frac_1 = RingSlicesPerLayer(
            fraction=1,
            rings_col="rings",
        )
        config_frac_2 = RingSlicesPerLayer(
            fraction=2,
            rings_col="rings",
        )

        cols_1 = config_frac_1.output_cols
        cols_2 = config_frac_2.output_cols

        assert len(cols_1) == 100  # All rings
        assert len(cols_2) == 50  # Half of rings
        assert len(cols_1) == 2 * len(cols_2)

    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    @pytest.mark.parametrize("dtype", ["list", "array"])
    def test_output_values_correctness(self, frame_factory, dtype):
        """Test that output values are correctly extracted from input."""

        config = RingSlicesPerLayer(
            fraction=1,  # Select all rings
            rings_col="rings",
        )

        input_list = [float(i) for i in range(100)]

        df = frame_factory({"rings": [input_list]})
        expected_result = pl.DataFrame(
            {f"rings.{i}": [input_list[i]] for i in range(len(input_list))}
        )

        if dtype == "array":
            df = df.select(pl.col("rings").list.to_array(100))

        result = config(df)
        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        assert_frame_equal(result, expected_result, check_dtype=True)


class TestUnnestStructs:

    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_unnest_single_level_struct(self, frame_factory):
        """Test unnesting a DataFrame/LazyFrame with a single level struct column."""
        input_data = {
            "id": [1, 2],
            "info": [{"name": "alice", "age": 30}, {"name": "bob", "age": 25}],
        }
        df = frame_factory(input_data)
        result = unnest_structs(df)

        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        expected = pl.DataFrame(
            {
                "id": [1, 2],
                "info.name": ["alice", "bob"],
                "info.age": [30, 25],
            }
        )
        assert_frame_equal(result, expected)


    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_unnest_deeply_nested_struct(self, frame_factory):
        """Test unnesting deeply nested struct columns."""
        input_data = {
            "a": [1],
            "nested": [
                {
                    "b": {"c": 10, "d": 20},
                    "e": {"f": {"g": 30}},
                }
            ],
        }
        df = frame_factory(input_data)
        result = unnest_structs(df)

        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        expected = pl.DataFrame(
            {
                "a": [1],
                "nested.b.c": [10],
                "nested.b.d": [20],
                "nested.e.f.g": [30],
            }
        )
        assert_frame_equal(result, expected)


    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_custom_separator(self, frame_factory):
        """Test unnesting structs with a custom separator."""
        input_data = {
            "user": [{"details": {"first": "John", "last": "Doe"}}],
        }
        df = frame_factory(input_data)
        result = unnest_structs(df, separator="_")

        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        expected = pl.DataFrame(
            {
                "user_details_first": ["John"],
                "user_details_last": ["Doe"],
            }
        )
        assert_frame_equal(result, expected)


    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_no_struct_columns(self, frame_factory):
        """Test that a DataFrame without struct columns is returned unchanged."""
        input_data = {
            "x": [1.0, 2.0],
            "y": ["a", "b"],
            "z": [[1, 2], [3, 4]],
        }
        df = frame_factory(input_data)
        result = unnest_structs(df)

        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        expected = pl.DataFrame(input_data)
        assert_frame_equal(result, expected)


    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_multiple_struct_columns(self, frame_factory):
        """Test unnesting multiple struct columns in the same DataFrame."""
        input_data = {
            "s1": [{"a": 1, "b": 2}],
            "val": [100],
            "s2": [{"c": 3, "d": 4}],
        }
        df = frame_factory(input_data)
        result = unnest_structs(df)

        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        expected = pl.DataFrame(
            {
                "s1.a": [1],
                "s1.b": [2],
                "val": [100],
                "s2.c": [3],
                "s2.d": [4],
            }
        )
        assert_frame_equal(result, expected)


    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_struct_with_null_values(self, frame_factory):
        """Test unnesting structs containing null values and fields."""
        schema = pl.Schema(
            {
                "id": pl.Int64,
                "info": pl.Struct([pl.Field("val", pl.Int64), pl.Field("text", pl.String)]),
            }
        )
        df = frame_factory(
            [
                {"id": 1, "info": {"val": 10, "text": "hello"}},
                {"id": 2, "info": None},
                {"id": 3, "info": {"val": None, "text": "world"}},
            ],
            schema=schema,
        )
        result = unnest_structs(df)

        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        expected = pl.DataFrame(
            [
                {"id": 1, "info.val": 10, "info.text": "hello"},
                {"id": 2, "info.val": None, "info.text": None},
                {"id": 3, "info.val": None, "info.text": "world"},
            ],
            schema=pl.Schema(
                {
                    "id": pl.Int64,
                    "info.val": pl.Int64,
                    "info.text": pl.String,
                }
            ),
        )
        assert_frame_equal(result, expected)


    def test_empty_struct_column(self):
        """Test handling of empty struct columns (struct with no fields)."""
        df = pl.DataFrame(
            schema={
                "id": pl.Int64,
                "empty_st": pl.Struct([]),
            }
        )
        result = unnest_structs(df)
        expected = pl.DataFrame(
            schema={
                "id": pl.Int64,
                "empty_st": pl.Struct([]),
            }
        )
        assert_frame_equal(result, expected)


    def test_returns_same_type(self):
        """Test that input type (DataFrame vs LazyFrame) is preserved in return value."""
        df = pl.DataFrame({"a": [1]})
        lf = pl.LazyFrame({"a": [1]})

        res_df = unnest_structs(df)
        res_lf = unnest_structs(lf)

        assert isinstance(res_df, pl.DataFrame)
        assert isinstance(res_lf, pl.LazyFrame)
