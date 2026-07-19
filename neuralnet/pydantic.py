from typing import Self
from pathlib import Path
from abc import ABC
from pydantic import BaseModel, ConfigDict
from pydantic.fields import PydanticUndefined


class YamlBaseModel(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        import yaml

        if isinstance(path, str):
            path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def pydantic_to_markdown_schema(model_class: type[BaseModel], indent: int = 0) -> str:
    """
    Generate a markdown-formatted schema description from a Pydantic model class.

    Recursively processes nested Pydantic models with proper indentation.

    Parameters
    ----------
    model_class : type[BaseModel]
        A Pydantic model class to generate schema for
    indent : int
        Current indentation level (used for recursion). Default is 0.

    Returns
    -------
    str
        A markdown-formatted string describing the model schema

    Examples
    --------
    >>> from neuralnet.models.vqat import VQATTrainingJob
    >>> schema = pydantic_to_markdown_schema(VQATTrainingJob)
    >>> print(schema)
    """
    lines = []
    next_indent_str = "  " * indent

    # Add docstring if available (only for root class, before the bullets)
    if indent == 0:
        docstring = model_class.__doc__
        if docstring:
            # Clean up the docstring (remove leading/trailing whitespace)
            docstring = docstring.strip()
            lines.append(docstring)

    # Get all fields from the model
    fields = model_class.model_fields

    for field_name, field_info in fields.items():
        # Get the type annotation
        field_type = field_info.annotation

        # Get a clean string representation of the type
        if hasattr(field_type, "__origin__"):
            # Handle generic types like Optional, Union, etc.
            type_str = str(field_type).replace("typing.", "")
        else:
            # Handle regular types
            type_str = getattr(field_type, "__name__", str(field_type))

        # Get the field description
        description = field_info.description or ""

        # Build default value suffix when a default is explicitly defined.
        default_suffix = ""
        if field_info.default is not PydanticUndefined:
            default_suffix = f" Default: {field_info.default!r}"
        elif field_info.default_factory is not None:
            factory_name = getattr(
                field_info.default_factory, "__name__", repr(field_info.default_factory)
            )
            default_suffix = f" Default factory: {factory_name}()"

        # Build the field line
        field_line = f"{next_indent_str}- {field_name} ({type_str})"
        if description:
            field_line += f": {description}"
        if default_suffix:
            if description:
                if description.endswith("."):
                    field_line += default_suffix
                else:
                    field_line += f".{default_suffix}"
            else:
                field_line += f":{default_suffix}"

        # If the field type is a Pydantic model, append its class docstring to the field line
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            nested_docstring = field_type.__doc__
            if nested_docstring:
                nested_docstring = nested_docstring.strip()
                # Append the nested class docstring to the field line (at the side)
                if description:
                    # If there's already a field description, append the class docstring
                    field_line += f" {nested_docstring}"
                else:
                    # If no field description, add the class docstring as the description
                    field_line += f": {nested_docstring}"

        lines.append(field_line)

        # If the field type is a Pydantic model, recursively add its nested fields
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            nested_schema = pydantic_to_markdown_schema(field_type, indent=indent + 2)
            lines.append(nested_schema)

    return "\n".join(lines)
