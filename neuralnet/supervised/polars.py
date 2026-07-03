from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
from . import get_logger


def compute_class_weight(
    classes: list[int],
    df: pl.LazyFrame | pl.DataFrame,
    label_col: str = "label",
) -> dict[int, float]:
    """
    Computes class weights based on the distribution of classes in the dataset.

    Args:
        classes (list[int]): List of unique class labels.
        df (LazyFrame | DataFrame): The dataset containing the class labels.

    Returns:
        dict[int, float]: A dictionary mapping each class label to its corresponding weight.
    """
    class_counts_df = (
        df.select(label_col).group_by(label_col).len(name="count").collect()
    )
    class_counts = {
        int(row[label_col]): row["count"]
        for row in class_counts_df.iter_rows(named=True)
    }
    logger = get_logger()
    for class_ in classes:
        if class_ not in class_counts:
            logger.warning(f"Class {class_} not found.")
            class_counts[class_] = 0

    total_samples = sum(class_counts.values())
    n_classes = len(classes)
    class_weights = {
        class_: total_samples / (n_classes * count) if count > 0 else 1.0
        for class_, count in class_counts.items()
    }
    return class_weights
