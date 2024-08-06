Scripts for the analysis of rivarly dataset with [svGPFA](https://github.com/joacorapela/svGPFA)
------------------------------------------------------------------------------------------------

# Provisional installation instructions

## get code
01. `mkdir svGPFAcode`
02. `cd svGPFAcode`
03. `git clone git@github.com:joacorapela/svGPFA.git`
04. `git clone git@github.com:joacorapela/gcnu_common.git`
05. `git clone git@github.com:joacorapela/svGPFA_rivarly.git`

## checkout specific branches is the svGPFA and gcnu_common repos
06. cd svGPFA
07. git checkout jaxTrickSameNSpikes
08. cd ../gcnu_common
09. git checkout jax1

## create conda environment and install dependencies
10. conda create -n svGPFAenv python=3.10
11. conda activaet svGPFAenv
12. cd ../svGPFA_rivarly
13. pip install -e../svGPFA
14. pip install -e../gcnu_common

## copy data file (saved in v6 Matlab format) to data subdirectory
15. cp /some_data_dir/Alfie_231024_200742_physical_switches_static_triplets_v6.mat data

## epoch dataset (not really epoching, but saving data in pickle format)
16. cd code/scripts
17. python doEpochSpikesTimes.py

## run the estimation script `doEstimateSVGPFA.py`

parameters of the estimation are provided in the ../../init/00000007_estimation_metaData.ini file
and as default command line arguments in the script `doEstimateSVGPFA.py`
(e.g., --n_latents, --common_n_ind_points)

request an interactive session in the cluster with one or more gpus and with large memory

18. srun -p a100 -t 3:00:00 --mem=200G --gres=gpu:1 --pty bash -i

## activate the environment svGPFA_rivarly in the cluster
19. conda activate svGPFA_rivarly

## customize optimization parameters

in the current implementation of the code the optimization parameters are hardcoded in
`doEstimateSVGPFA.py`. You may want to change maxiter or tol in the dictionary optim_params
(the maximum number of iteration and the convergence tolerance for the optimization)

## run `doEstimateSVGPFA.py`
20. python doEstimateSVGPFA.py

the optimization result is a dictionary with keys
params: svGPFA parameters
lower_bound_hist: list of (increasing) lower bound values achieved at each iteration
elapse_time_hist: list of elapsed times for each iteration
state: LBFGS optimization state (used only for debugging)

## Notes

1. it may be challenging to fit too many trials and neuron in the GPU. I was able to fit all 498 trials and all neurons with mean firing rate larger than 5 Hz in a GPU with 200 Gb of memory.

2. Please let me know if you have any problem/question.
