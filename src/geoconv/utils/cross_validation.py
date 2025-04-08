from geoconv.utils.data_generator import preprocessed_properties_generator


def get_folds_and_splits(dataset_path, amount_folds):
    """Computes dataset folds and splits for cross-validation.

    Parameters
    ----------
    dataset_path: str
        Path to the preprocessed dataset.
    amount_folds: int
        The amount of folds to consider.

    Returns
    -------
    dict, dict:
        The first dictionary contains the folds. The second dictionary contains the splits.
    """
    # Determine dataset length
    ppg = preprocessed_properties_generator(dataset_path)
    dataset_size = next(ppg)["preprocessed_shapes"]

    # Determine folds
    chunk = dataset_size // amount_folds
    dataset_folds = {-1: list(range(0, dataset_size))}
    for fold in range(amount_folds):
        if fold < amount_folds - 1:
            dataset_folds[fold] = list(range(fold * chunk, fold * chunk + chunk))
        else:
            dataset_folds[fold] = list(range(fold * chunk, dataset_size))

    # Determine splits
    fold_indices = list(range(amount_folds))
    dataset_split_indices = {
        split: fold_indices[:split] + fold_indices[split + 1 :]
        for split in fold_indices
    }
    dataset_splits = {}
    for key, fold_indices in dataset_split_indices.items():
        dataset_splits[key] = [
            shape_idx for idx in fold_indices for shape_idx in dataset_folds[idx]
        ]

    return dataset_folds, dataset_splits
