import pandas as pd
import numpy as np
import json


def process_logs(csv_files, file_name):
    """Averages final training logs over multiple CSV-files.

    Parameters
    ----------
    csv_files: list
        A list of paths to CSV-training-log-files. Of each given file, the last, i.e., the final epoch, row will be read
        and averaged over all given CSV-files.
    file_name: str
        A path where the averages shall be saved.
    """
    # Parse available statistics
    captured_statistics = pd.read_csv(csv_files[0]).columns

    # Init stats dictionary
    stats_dict = {}

    # Iterate over all available statistics
    for statistic in captured_statistics:
        stats_dict[statistic] = []
        # Iterate over all available CSV-files
        for csv_filename in csv_files:
            csv_df = pd.read_csv(csv_filename)
            # Capture final statistic in CSV-file
            stats_dict[statistic].append(float(csv_df[statistic].iloc[-1]))
        # Average current statistic
        stats_dict[statistic] = {
            "mean": np.mean(stats_dict[statistic]),
            "variance": np.var(stats_dict[statistic]),
        }

    # Log stats-dict into *.json-file
    file_name = f"{file_name}.json" if file_name[-4:] != "json" else file_name
    with open(file_name, "w") as training_logs_file:
        json.dump(stats_dict, training_logs_file, indent=4)
