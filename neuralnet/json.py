from collections.abc import Sequence
from pydantic import JsonValue, BaseModel
import json


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def cast_to_json_value(data) -> JsonValue:

    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    elif isinstance(data, dict):
        return {str(key): cast_to_json_value(value) for key, value in data.items()}
    elif isinstance(data, Sequence):
        return [cast_to_json_value(item) for item in data]
    elif isinstance(data, BaseModel):
        return json.loads(data.model_dump_json())
    else:
        raise TypeError(f"Unsupported type for JSON serialization: {type(data)}")
