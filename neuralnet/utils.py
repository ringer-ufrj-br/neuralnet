from typing import Any, Iterable, Iterator
from collections.abc import Mapping, Sequence, Iterable
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import PydanticUndefined


def walk_paths(
    paths: str | Path | Iterable[str | Path], file_ext: str, dev: bool = False
) -> Iterator[Path]:
    """
    Generator that opens all directories in an iterator for
    a specific file extension. This is useful for script cases where
    an user can pass a mix of directories and filepaths.

    Parameters
    ----------
    paths : str | Path | Iterable[str | Path]
        A single path or an iterable of paths. These can be directories or
        file paths. If a directory is provided, it will search recursively
        for files with the specified file extension.
    file_ext : str
        The desired file extension to look for
    dev: bool
        If True, the function will yield just the first file found

    Yields
    ------
    Path
        The path to a file

    Raises
    ------
    ValueError
        Raised if there is a file that does not have file_ext as its extension
    """
    if isinstance(paths, str):
        paths = [Path(paths)]
    elif isinstance(paths, Path):
        paths = [paths]
    i = 0
    for ipath in paths:
        if ipath.is_file():
            if ipath.suffix != f".{file_ext}":
                raise ValueError(
                    f"File {ipath} does not have the expected extension .{file_ext}"
                )
            yield ipath
            i += 1
            if dev and i > 0:
                break
        else:
            for filepath in ipath.glob(f"**/*.{file_ext}"):
                yield filepath
                i += 1
                if dev and i > 0:
                    break


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

def traverse(
    d: Mapping[Any, Any] | Iterable[Any], parent_key: str = ""
) -> Iterator[tuple[str, Any]]:
    """
    Recursively traverse a mapping or iterable of objects and yield key-value pairs.

    Parameters
    ----------
    d : dict | Iterable
        The mapping or iterable to traverse.
    parent_key : str
        The base key to prepend to each key in the nested mappings.

    Yields
    ------
    tuple[str, Any]
        A tuple containing the full key path and its corresponding value.
    """
    if isinstance(d, Mapping):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, (Mapping, Iterable)) and not isinstance(v, (str, bytes)):
                yield from traverse(v, new_key)
            else:
                yield new_key, v
    elif isinstance(d, Iterable) and not isinstance(d, (str, bytes)):
        for i, item in enumerate(d):
            item_key = f"{parent_key}.{i}" if parent_key else str(i)
            if isinstance(item, (Mapping, Iterable)) and not isinstance(item, (str, bytes)):
                yield from traverse(item, item_key)
            else:
                yield item_key, item
    else:
        raise ValueError("Input must be a mapping or an iterable of objects")
