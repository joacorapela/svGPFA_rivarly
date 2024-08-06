
import sys
import os
import numpy as np
import scipy.io
import argparse
import pickle

import gcnu_common.utils
import svGPFA.plot.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../data/Alfie_231024_200742_physical_switches_static_triplets_v6.mat")
    parser.add_argument("--fig_filename_pattern",
                        help=("figure filename pattern"),
                        type=str,
                        default=("../../figures/spikes_rates_{:s}.{:s}"))
    args = parser.parse_args()

    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    descriptor, _ = os.path.splitext(os.path.basename(data_filename))
    descriptor = descriptor[:-3] # remove _v6

    print(f"loading {data_filename}")
    mat = scipy.io.loadmat(data_filename)
    print(f"done loading {data_filename}")

    n_trials = mat["data"]["spike_times"][0,0].shape[0]
    n_clusters = mat["data"]["spike_times"][0,0][0,0].shape[1]
    spikes_times = [[None for n in range(n_clusters)] for r in range(n_trials)]
    Fs = 1e3
    trials_start_times = [0.0 for r in range(n_trials)]
    trials_end_times = \
        mat["data"]["trial_data"][0,0]["trial_durations"][0,0][0,:]/Fs
    for r in range(n_trials):
        for n in range(n_clusters):
            spikes_times[r][n] = \
                np.where(mat["data"]["binned_response"][0,0][r,0][:,n])[0]/Fs

    trials_durations = np.array([trials_end_times[r] - trials_start_times[r]
                                 for r in range(n_trials)])
    spikes_rates_allTrials_allNeurons = \
        gcnu_common.utils.neural_data_analysis.\
        getSpikesRatesAllTrialsAllNeurons(
            spikes_times=spikes_times, trials_durations=trials_durations)
    averaged_spikes_rates = spikes_rates_allTrials_allNeurons.mean(axis=0)

    trials_ids = np.arange(n_trials)
    unit_probes_index = mat["data"]["unit_probes_index"][0,0][0,:]
    regions = [region.item() for region in mat["data"]["regions"][0,0][0, :]]
    types = [aType.item() for aType in mat["data"]["unit_type"][0,0][0, :]]
    clusters_ids = [f"cluster: {n}, region: {regions[unit_probes_index[n]-1]}, type: {types[n]}, ASR: {averaged_spikes_rates[n]:.0f}"
                   for n in range(n_clusters)]
    fig = svGPFA.plot.plotUtilsPlotly.\
        getPlotSpikesRatesAllTrialsAllClusters(
            spikes_rates=spikes_rates_allTrials_allNeurons,
            trials_ids=trials_ids, clusters_ids=clusters_ids)

    fig.write_image(fig_filename_pattern.format(descriptor, "png"))
    fig.write_html(fig_filename_pattern.format(descriptor, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
