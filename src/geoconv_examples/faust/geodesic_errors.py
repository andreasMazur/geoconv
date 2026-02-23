from geoconv.preprocessing.distance_computation import calculate_local_charts

from tqdm import tqdm

import numpy as np


def avg_geodesic_error(model, ref_shape, dataset, distance_calc_method="hdm", processes=1):
    # Prepare geodesic distances
    geo_distances = calculate_local_charts(
        ref_shape,
        method=distance_calc_method,
        processes=processes,
        max_radius=np.inf,
        calculate_angle=False,
        process_description="Calculating geodesic distances on reference shape.",
    )

    geodesic_errors = np.zeros((0,))
    for inputs, ground_truth in tqdm(dataset, postfix="Calculating geodesic errors.."):
        # Sort distances according to permutation
        query_order_gd = geo_distances[ground_truth[0].numpy().astype(np.int32)]

        # Compute prediction
        y_pred = model(inputs)[0]
        y_pred = y_pred.numpy().argmax(axis=-1)

        # Get geodesic errors
        query_order_gd = query_order_gd[np.arange(y_pred.shape[0]), y_pred]

        # Append to prior retrieved errors
        geodesic_errors = np.concatenate([geodesic_errors, query_order_gd], axis=0)
    return geodesic_errors
