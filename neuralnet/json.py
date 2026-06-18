from collections.abc import Sequence
from pydantic import JsonValue, BaseModel
import json
import numpy as np
from datetime import datetime


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
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.str_):
        return str(data)
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        raise TypeError(f"Unsupported type for JSON serialization: {type(data)}")
