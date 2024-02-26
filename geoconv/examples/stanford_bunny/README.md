# Example script for a pre-processing, hyperparameter tuning and training pipeline

This demo is meant to illustrate a quick insight into intermediate pre-processing results, at the hand of the 
Stanford-bunny.

You can call the demo by writing a script that calls `preprocess_demo`. E.g:

```python
from geoconv.examples.stanford_bunny.preprocess_demo import preprocess_demo

if __name__ == "__main__":
    # Example path to Stanford-bunny:
    path_to_stanford_bunny = "/home/user/geoconv/geoconv/examples/stanford_bunny/data/bun_zipper.ply"
    preprocess_demo(path_to_stanford_bunny, n_radial=5, n_angular=8, processes=1)
```