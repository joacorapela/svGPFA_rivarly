
import sys
import os
import argparse
import configparser
import numpy as np
import scipy
import pickle

import rivarlyUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../data/Alfie_231024_200742_physical_switches_static_triplets_v6.mat")
    parser.add_argument("--results_filename_pattern",
                        help="results filename pattern",
                        type=str,
                        default="../../results/epochedSpikes_{:s}.{:s}")
    args = parser.parse_args()

    data_filename = args.data_filename
    results_filename_pattern = args.results_filename_pattern

    descriptor, _ = os.path.splitext(os.path.basename(data_filename))
    descriptor = descriptor[:-3] # remove _v6

    print(f"loading {data_filename}")
    mat = scipy.io.loadmat(data_filename)
    print(f"done loading {data_filename}")

    n_trials, n_units, spikes_times = rivarlyUtils.getSpikesTimes(mat=mat)

    Fs = 1e3
    trials_start_times = [0.0 for r in range(n_trials)]
    trials_end_times = \
        mat["data"]["trial_data"][0,0]["trial_durations"][0,0][0,:]/Fs
    unit_probes_index = mat["data"]["unit_probes_index"][0,0][0,:]
    unit_id = mat["data"]["unit_id"][0,0][0,:]
    unit_type = [mat["data"]["unit_type"][0,0][0,n][0] for n in range(n_units)]
    regions = [region.item() for region in mat["data"]["regions"][0,0][0, :]]
    types = [aType.item() for aType in mat["data"]["unit_type"][0,0][0, :]]

    epoch_config = configparser.ConfigParser()
    epoch_config["params"] = {
        "data_filename": data_filename,
    }
    metadata_filename = results_filename_pattern.format(descriptor, "metadata")
    with open(metadata_filename, "w") as f:
        epoch_config.write(f)
    print(f"Saved {metadata_filename}")

    results = {"spikes_times": spikes_times,
               "trials_start_times": trials_start_times,
               "trials_end_times": trials_end_times,
               "unit_probes_index": unit_probes_index,
               "unit_id": unit_id,
               "unit_type": unit_type,
               "region": regions,
               "type": types,
              }
    results_filename = results_filename_pattern.format(descriptor, "pickle")
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {results_filename}")


if __name__ == "__main__":
    main(sys.argv)
