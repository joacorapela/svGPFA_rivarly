import sys
import os
import time
import random
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse
import configparser
import warnings

import gcnu_common.utils.neural_data_analysis
import gcnu_common.utils.config_dict
import svGPFA.stats.em
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils

import rivarlyUtils

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_init_number", help="estimation init number",
                        type=int, default=7)
    parser.add_argument("--n_latents", help="number of latent processes",
                        type=int, default=5)
    parser.add_argument("--common_n_ind_points",
                        help="common number of inducing points",
                        type=int, default=8)
    parser.add_argument("--profile",
                        help="use this option if you want to profile svGPFA.maximize()",
                        action="store_true")
    parser.add_argument("--epoched_spikes_times_filename_pattern",
                        help="epoched spikes times filename pattern",
                        type=str,
                        default="../../results/epochedSpikes_Alfie_231024_200742_physical_switches_static_triplets.{:s}")
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../init/{:08d}_estimation_metaData.ini")
    parser.add_argument("--estim_res_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimation_metaData.ini")
    parser.add_argument("--profiling_info_filename_pattern",
                        help="profiling information filename pattern",
                        type=str,
                        default="../../results/{:08d}_profiling_info.txt")
    parser.add_argument("--trials_ids_filename", help="trials ids filename",
                        type=str, default="../../init/trialsIDs_0_497.csv")
    parser.add_argument("--clusters_ids_filename", help="clusters ids filename",
                        type=str, default="../../init/clustersIDs_0_1024.csv")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimatedModel.pickle")
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str)
    args = parser.parse_args()

    est_init_number = args.est_init_number
    n_latents = args.n_latents
    common_n_ind_points = args.common_n_ind_points
    profile = args.profile
    epoched_spikes_times_filename_pattern = args.epoched_spikes_times_filename_pattern
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    estim_res_metadata_filename_pattern = \
        args.estim_res_metadata_filename_pattern
    profiling_info_filename_pattern = args.profiling_info_filename_pattern
    trials_ids_filename = args.trials_ids_filename
    clusters_ids_filename = args.clusters_ids_filename
    model_save_filename_pattern = args.model_save_filename_pattern

    est_init_config_filename = est_init_config_filename_pattern.format(
        est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)

    min_neuron_trials_avg_firing_rate = float(est_init_config["data_params"]["min_neuron_trials_avg_firing_rate"])

    # get spike_times
    epoched_spikes_times_filename = \
        epoched_spikes_times_filename_pattern.format("pickle")
    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]
    trials_start_times = load_res["trials_start_times"]
    trials_end_times = load_res["trials_end_times"]

    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    trials_ids = jnp.arange(n_trials)

    # subset selected_clusters_ids
    selected_clusters_ids = np.genfromtxt(clusters_ids_filename,
                                          dtype=np.uint64)
    clusters_ids = np.arange(n_neurons)
    spikes_times = rivarlyUtils.subset_clusters_ids_data(
        selected_clusters_ids=selected_clusters_ids,
        clusters_ids=clusters_ids,
        spikes_times=spikes_times,
    )

    # subset selected_trials_ids
    selected_trials_ids = np.genfromtxt(trials_ids_filename, dtype=np.uint64)
    spikes_times, trials_start_times, trials_end_times = \
        rivarlyUtils.subset_trials_ids_data(
            selected_trials_ids=selected_trials_ids,
            trials_ids=trials_ids,
            spikes_times=spikes_times,
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)

    # breakpoint()

    n_trials = len(spikes_times)
    # trials_indices = np.arange(n_trials)
    n_neurons = len(spikes_times[0])
    neurons_indices = jnp.arange(n_neurons).tolist()

    trials_durations = [trials_end_times[i] - trials_start_times[i]
                        for i in range(n_trials)]
    spikes_times, neurons_indices = \
        gcnu_common.utils.neural_data_analysis.removeUnitsWithLessTrialAveragedFiringRateThanThr(
            spikes_times=spikes_times, neurons_indices=neurons_indices,
            trials_durations=trials_durations,
            min_neuron_trials_avg_firing_rate=min_neuron_trials_avg_firing_rate)
    clusters_ids = [clusters_ids[i] for i in neurons_indices]

    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    # breakpoint()

    args_info = svGPFA.utils.initUtils.getArgsInfo()
    #    build dynamic parameter specifications
    dynamic_params_spec = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=vars(args),
        args_info=args_info)
    #   build config file parameters specification
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=est_init_config).get_dict()
    config_file_params_spec = \
        svGPFA.utils.initUtils.getParamsDictFromStringsDict(
            n_latents=n_latents, n_trials=n_trials,
            strings_dict=strings_dict, args_info=args_info)
    #    build default parameter specificiations
    default_params_spec = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_latents=n_latents,
        common_n_ind_points=common_n_ind_points)
    #    finally, get the parameters from the dynamic,
    #    configuration file and default parameter specifications
    params, kernels_types, = \
        svGPFA.utils.initUtils.getParamsAndKernelsTypes(
            n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times,
            dynamic_params_spec=dynamic_params_spec,
            config_file_params_spec=config_file_params_spec)
            # config_file_params_spec=config_file_params_spec,
            # default_params_spec=default_params_spec)

    kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]

    # build modelSaveFilename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estim_res_metadata_filename = \
            estim_res_metadata_filename_pattern.format(estResNumber)
        if not os.path.exists(estim_res_metadata_filename):
            estPrefixUsed = False
    modelSaveFilename = model_save_filename_pattern.format(estResNumber)
    if profile:
        profiling_info_filename_pattern = \
            profiling_info_filename_pattern.format(estResNumber)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params0)

    spikes_times_array, valid_spikes_times_mask = \
        svGPFA.utils.miscUtils.buildSpikesTimesArray(spikes_times=spikes_times)

    leg_quad_weights = params["ell_calculation_params"]["leg_quad_weights"]
    leg_quad_points = params["ell_calculation_params"]["leg_quad_points"]
    qMu0 = params["initial_params"]["posterior_on_latents"]["posterior_on_ind_points"]["mean"].squeeze()
    variational_chol_vecs = params["initial_params"]["posterior_on_latents"]["posterior_on_ind_points"]["cholVecs"]
    C = params["initial_params"]["embedding"]["C0"]
    d = params["initial_params"]["embedding"]["d0"]
    Z0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["inducing_points_locs0"]

    # save estimated values
    estim_res_config = configparser.ConfigParser()
    estim_res_config["data_params"] = {
        "trials_ids": selected_trials_ids,
        "neurons_indices": neurons_indices,
        "clusters_ids": selected_clusters_ids,
        "nLatents": n_latents,
        "common_n_ind_points": common_n_ind_points,
        # "max_trial_duration": max_trial_duration,
        "min_neuron_trials_avg_firing_rate": min_neuron_trials_avg_firing_rate,
        "epoched_spikes_times_filename": epoched_spikes_times_filename,
    }
    estim_res_config["estimation_params"] = {"est_init_number":
                                             est_init_number}
    with open(estim_res_metadata_filename, "w") as f:
        estim_res_config.write(f)
    print(f"Saved {estim_res_metadata_filename}")

    em = svGPFA.stats.em.EM_JAXopt
    em.init(spikesTimesArray=spikes_times_array,
            validSpikesTimesMask=valid_spikes_times_mask, kernels=kernels,
            legQuadPoints=leg_quad_points, legQuadWeights=leg_quad_weights,
            reg_param=params["optim_params"]["prior_cov_reg_param"])

    params0 = dict(
        variational_mean=qMu0,
        variational_chol_vecs=variational_chol_vecs,
        C=C,
        d=d,
        kernels_params=kernels_params0,
        ind_points_locs=Z0,
    )
    optim_params = dict(
        # maxiter=params["optim_params"]["em_max_iter"],
        maxiter=10000,
        tol=1e-6,
        max_stepsize=1.0,
        jit=True,
        # verbose=True,
    )
    optim_params_ECM = dict(
        n_em_iterations=50,
        tol = 1e-4,
        variational_estimate=True,
        variational_params=dict(
            jit=True,
            tol=1e-2,
            # options={"gtol": 1e-10, "maxcor": 100},
            maxiter=100,
            # maxls=20,
            max_stepsize=0.1,
            history_size=100,
        ),
        preIntensity_estimate=True,
        preIntensity_params=dict(
            jit=True,
            tol=1e-2,
            # options={"gtol": 1e-10, "maxcor": 100},
            maxiter=100,
            # maxls=20,
            max_stepsize=0.01,
            history_size=100,
        ),
        kernels_estimate=True,
        kernels_params=dict(
            jit=True,
            tol=1e-2,
            # options={"gtol": 1e-10, "maxcor": 100},
            maxiter=100,
            # maxls=20,
            max_stepsize=0.1,
            history_size=100,
        ),
        indpointslocs_estimate=True,
        indpointslocs_params=dict(
            jit=True,
            tol=1e-2,
            # options={"gtol": 1e-10, "maxcor": 100},
            maxiter=100,
            # maxls=20,
            max_stepsize=0.1,
            history_size=100,
        ),
    )

    start_time = time.time()
    # res = em.maximize(params0=params0, optim_params=optim_params)
    res = em.maximizeInSteps(params0=params0, optim_params=optim_params)
    # res = em.maximize_jaxopt_scipy(params0=params0, optim_params=optim_params)
    # res = em.maximizeECM(params0=params0, optim_params=optim_params_ECM)
    elapsed_time = time.time() - start_time
    print(f"elapsed time={elapsed_time}")

    resultsToSave = res
    with open(modelSaveFilename, "wb") as f:
        pickle.dump(resultsToSave, f)
        print("Saved results to {:s}".format(modelSaveFilename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
