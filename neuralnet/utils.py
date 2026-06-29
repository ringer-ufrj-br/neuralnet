from typing import Any, Iterable, Iterator
from collections.abc import Mapping, Sequence
from pathlib import Path


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


def traverse(
    d: Mapping[Any, Any] | Iterable[Any],
    parent_key: str = "",
    include_sequences: bool = False,
) -> Iterator[tuple[str, Any]]:
    """
    Recursively traverse a mapping or iterable of objects and yield key-value pairs.

    Parameters
    ----------
    d : dict | Iterable
        The mapping or iterable to traverse.
    parent_key : str
        The base key to prepend to each key in the nested mappings.
    include_sequences: bool
        If True, the function will also traverse sequences (like lists and tuples).
        If False, sequences will be treated as leaf nodes and not traversed.

    Yields
    ------
    tuple[str, Any]
        A tuple containing the full key path and its corresponding value.
    """
    if include_sequences:
        traverse_types = (Mapping, Sequence)
    else:
        traverse_types = (Mapping,)
    if isinstance(d, Mapping):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, traverse_types) and not isinstance(v, (str, bytes)):
                yield from traverse(v, new_key)
            else:
                yield new_key, v
    elif isinstance(d, Iterable) and not isinstance(d, (str, bytes)):
        for i, item in enumerate(d):
            item_key = f"{parent_key}.{i}" if parent_key else str(i)
            if isinstance(item, traverse_types) and not isinstance(item, (str, bytes)):
                yield from traverse(item, item_key)
            else:
                yield item_key, item
    else:
        raise ValueError("Input must be a mapping or an iterable of objects")


def unflatten_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """
    Unflatten a mapping with dot-separated keys into a nested dictionary.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        A mapping with dot-separated keys.

    Returns
    -------
    dict[str, Any]
        A nested dictionary representation of the input mapping.
    """
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part in d and not isinstance(d[part], dict):
                raise KeyError(f"Duplicate key found: {key}")
            elif part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def get_ring_slices_per_layer(fraction: int) -> list[int]:
    # We select 1/fraction of rings in each layer
    # pre-sample - 8 rings
    # EM1 - 64 rings
    # EM2 - 8 rings
    # EM3 - 8 rings
    # Had1 - 4 rings
    # Had2 - 4 rings
    # Had3 - 4 rings
    rings_indexes = []
    # rings presmaple
    rings_indexes += list(range(8 // fraction))

    # EM1 list
    sum_rings = 8
    rings_indexes += list(range(sum_rings, sum_rings + (64 // fraction)))

    # EM2 list
    sum_rings = 8 + 64
    rings_indexes += list(range(sum_rings, sum_rings + (8 // fraction)))

    # EM3 list
    sum_rings = 8 + 64 + 8
    rings_indexes += list(range(sum_rings, sum_rings + (8 // fraction)))

    # HAD1 list
    sum_rings = 8 + 64 + 8 + 8
    rings_indexes += list(range(sum_rings, sum_rings + (4 // fraction)))

    # HAD2 list
    sum_rings = 8 + 64 + 8 + 8 + 4
    rings_indexes += list(range(sum_rings, sum_rings + (4 // fraction)))

    # HAD3 list
    sum_rings = 8 + 64 + 8 + 8 + 4 + 4
    rings_indexes += list(range(sum_rings, sum_rings + (4 // fraction)))

    return rings_indexes
