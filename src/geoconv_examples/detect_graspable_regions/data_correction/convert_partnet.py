from geoconv_examples.detect_graspable_regions.data.dataset import processed_data_generator
from geoconv_examples.improving_segmentation.data.convert_dataset import convert_dataset_deepview

from pathlib import Path


def convert_partnet(old_data_path, new_data_path, csv_path, label_changes_path=None):
    """Integrates label corrections into the old dataset, thereby creating a new one."""
    # Integrate DeepView-corrected labels into the dataset
    old_data_path = f"{old_data_path}.zip"
    if not Path(new_data_path).is_file():
        old_dataset = processed_data_generator(path_to_zip=old_data_path, set_type=3,)
        convert_dataset_deepview(
            csv_path=csv_path,
            old_dataset=old_dataset,
            new_dataset_path=new_data_path,
            signals_are_coordinates=True,
            label_changes_path=label_changes_path
        )
    else:
        print(f"Found converted dataset file: '{new_data_path}'. Skipping preprocessing.")
