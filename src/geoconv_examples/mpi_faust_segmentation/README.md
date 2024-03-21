# Example: FAUST segmentation

```python
from src.geoconv_examples.mpi_faust_segmentation.data.segment_meshes import compute_seg_labels

if __name__ == "__main__":
    registration_path = "/home/andreas/Uni/datasets/MPI-FAUST/training/registrations"
    labels_path = "./segmentation_labels"
    compute_seg_labels(registration_path, labels_path, verbose=True)
```