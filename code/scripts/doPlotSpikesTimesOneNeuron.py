
import sys
import argparse
import numpy as np
import scipy

import gcnu_common.utils
import svGPFA.plot.plotUtilsPlotly

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../data/Alfie_231024_200742_physical_switches_static_triplets_v6.mat")
    parser.add_argument("--neuron_index", type=int,
                        help="neuron to plot spikes for all trials",
                        default=0)
    parser.add_argument("--trials_indices", type=str,
                        help="trials indices to analyze", default=None)
                        # default="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]")
    parser.add_argument("--title_pattern", type=str, help="title pattern",
                        default="Cluster {:d}, Region: {:s}, Type: {:s}, Average Spike Rate (Hz): {:.02f}")
    parser.add_argument("--trial_start_colour", type=str,
                        help="trial start colour", default="black")
    parser.add_argument("--trial_end_colour", type=str,
                        help="trial end colour", default="red")
    parser.add_argument("--vline_width", type=int, help="vertical line width",
                        default=3)
    parser.add_argument("--vline_style", type=str, help="vertical line style",
                        default="solid")
    parser.add_argument("--xlabel", type=str, help="x-axis label",
                        default="Time (sec)")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure spikes for one neuron filename pattern",
                        default="../../figures/spikesTimes_neuron{:d}.{:s}")
    args = parser.parse_args()

    data_filename = args.data_filename
    neuron_index = args.neuron_index
    if args.trials_indices is not None:
        trials_indices = [int(str) for str in args.trials_indices[1:-1].split(",")]
    else:
        trials_indices = None
    title_pattern = args.title_pattern
    trial_start_colour = args.trial_start_colour
    trial_end_colour = args.trial_end_colour
    vline_width = args.vline_width
    vline_style = args.vline_style
    xlabel = args.xlabel
    fig_filename_pattern = args.fig_filename_pattern

    print(f"loading {data_filename}")
    mat = scipy.io.loadmat(data_filename)
    print(f"done loading {data_filename}")

    n_trials = mat["data"]["spike_times"][0,0].shape[0]
    n_neurons = mat["data"]["spike_times"][0,0][0,0].shape[1]
    spikes_times = [[None for n in range(n_neurons)] for r in range(n_trials)]
    Fs = 1e3
    trials_start_times = [0.0 for r in range(n_trials)]
    trials_end_times = \
        mat["data"]["trial_data"][0,0]["trial_durations"][0,0][0,:]/Fs
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]
    for r in range(n_trials):
        marked_events_times[r] = [trials_start_times[r], trials_end_times[r]]
        marked_events_colors[r] = [trial_start_colour, trial_end_colour]
        marked_events_markers[r] = ["cross", "cross"]
        for n in range(n_neurons):
            spikes_times[r][n] = \
                np.where(mat["data"]["binned_response"][0,0][r,0][:,n])[0]/Fs

    trials_durations = np.array([trials_end_times[r] - trials_start_times[r]
                                 for r in range(n_trials)])
    spikes_rates_allTrials_allNeurons = \
        gcnu_common.utils.neural_data_analysis.\
        getSpikesRatesAllTrialsAllNeurons(
            spikes_times=spikes_times, trials_durations=trials_durations)
    averaged_spikes_rates = spikes_rates_allTrials_allNeurons.mean(axis=0)

    unit_probes_index = mat["data"]["unit_probes_index"][0,0][0,:]
    regions = [region.item() for region in mat["data"]["regions"][0,0][0, :]]
    types = [aType.item() for aType in mat["data"]["unit_type"][0,0][0, :]]
    title= title_pattern.format(neuron_index,
                                regions[unit_probes_index[neuron_index]-1],
                                types[neuron_index],
                                averaged_spikes_rates[neuron_index])
    fig = svGPFA.plot.plotUtilsPlotly.getSpikesTimesPlotOneNeuron(
        spikes_times=spikes_times,
        neuron_index=neuron_index,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        title=title, xlabel=xlabel, ylabel="Trial")
    png_fig_filename = fig_filename_pattern.format(neuron_index, "png")
    fig.write_image(png_fig_filename)
    html_fig_filename = fig_filename_pattern.format(neuron_index, "html")
    fig.write_html(html_fig_filename)

    fig.show()
    breakpoint()

if __name__=="__main__":
    main(sys.argv)
