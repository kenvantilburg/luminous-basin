# Code

## Yellin method preparation

0. Run [script_00_yellin_mc_vols.py](script_00_yellin_mc_vols.py) with syntax `script_00_yellin_mc_vols.py i` with `i` running over a large set of integers (default: 200). This integer runs over the elements of `list_mu`, and is the number of expected counts inside the unit cuboid. This will generate a large number (`N_MC`) of Monte Carlo volumes for each `mu` in the specified output directory. They are written to a directory called `dir_ceph`, and are to be used for the later Yellin analysis. (Only needs to be done once but takes a long time.)

1. Run [script_01_yellin_cmax_bar.py](script_01_yellin_cmax_bar.py) with syntax `script_01_yellin_cmax_bar.py i` with `i` running over the same set of integers. This will compute C_max_bar as a function of mu based on the Monte Carlo, save the result in a csv file in the `dir_mc` folder, and produce a figure in the [figures](../figures/) folder. (This only needs to be done once.) 

The notebook `nb_01_yellin_cmax_bar.ipynb` collects the MC results into a plot.

## Yellin/Poisson analysis scripts on mocks (with fiducial solar position)
For the next 3 steps, companion notebooks [nb_02_yellin_proj_fid_mock.ipynb](nb_02_yellin_proj_fid_mock.ipynb), [nb_03_yellin_proj_fid_mock.ipynb](nb_03_yellin_proj_fid_mock.ipynb), and [nb_04_yellin_cmax_mock.ipynb](nb_04_yellin_cmax_mock.ipynb), with similar functionality as their eponymous scripts, may be used for debugging/inspection purposes.

2. Run [script_02_yellin_proj_fid_mock.py](script_02_yellin_proj_fid_mock.py) with syntax `script_02_yellin_proj_fid_mock.py jmock im` to project photon events from mock data onto the 4D unit cuboid. For the mocks, `jmock` denotes an integer labeling the mock data set. The integer `im` runs over the different axion masses. 

3. Run [script_03_yellin_vols_mock.py](script_03_yellin_vols_mock.py) with syntax `script_03_yellin_vols_mock.py jmock im k` to compute the Yellin volumes on the projection files from the previous step. 

4. Run [script_04_yellin_cmax_mock.py](script_04_yellin_cmax_mock.py) with syntax `script_04_yellin_cmax_mock.py jmock im k` to compute C_max values based on the CDFs from the Monte Carlo run (step 0) and the computed volumes (step 3) on the mock data. This produces a results file called `file_N_sig_lim` in the `dir_res` folder.

## Likelihood analysis script on mocks (marginalizing over solar position)
5. Run [script_05_mcmc_mock.py](script_05_mcmc_mock.py) with syntax `script_05_mcmc_mock.py jmock im` to run an MCMC on the full mock data set at each mass value. This producees a results file with filename specified by `file_LL_lim` in the `dir_res` folder. The MCMC chains are stored for post-analysis purposes in the `dir_mcmc` folder. The notebook [nb_15_likelihood_mock.ipynb](nb_15_likelihood_mock.ipynb) is an interactive version of the script that can run the MCMC for a single mass value.

## Post-analysis on mocks
6. The notebook [nb_06_limit_mock.py](nb_06_limit_mock.py) collects the results from steps 2--5, and synthesizes them into limit figures.

## Yellin/Poisson analysis scripts on data (marginalizing over solar position)
12. Run [script_12_yellin_proj_data.py](script_12_yellin_proj_data.py) with syntax `script_12_yellin_proj_data.py ip im` to project photon events from the real data onto the 4D unit cuboid. The integer `ip = 1,...,317` denotes an integer labeling the projection; each projection has a different solar position. The integer `im` runs over the different axion masses.

13. Run [script_13_yellin_vols_data.py](script_13_yellin_vols_data.py) with syntax `script_13_yellin_vols_data.py ip im k` to compute the Yellin volumes on the projection files from the previous step. The integer `k` labels the number of bins used in each of the 4 dimensions of the unit cuboid.

14. Run [script_14_yellin_cmax_data.py](script_14_yellin_cmax_data.py) with syntax `script_14_yellin_cmax_data.py ip im k` to compute C_max values based on the CDFs from the Monte Carlo run (step 0) and the computed volumes (step 13) on the mock data. This produces a results file called `file_N_sig_lim` in the `dir_res` folder.

## Likelihood analysis script on mocks (marginalizing over solar position)
15. Run [script_15_mcmc_data.py](script_15_mcmc_data.py) with syntax `script_15_mcmc_data.py im` to run an MCMC on the real data set at each mass value. This producees a results file with filename specified by `file_LL_lim` in the `dir_res` folder. The MCMC chains are stored for post-analysis purposes in the `dir_mcmc` folder. The notebook [nb_15_likelihood_data.ipynb](nb_15_likelihood_data.ipynb) is an interactive version of the script that can run the MCMC for a single mass value.

## Post-analysis on data
16. The notebook [nb_16_lim_data.ipynb](nb_16_lim_data.ipynb) collects the resutls from steps 12--15, and synthesizes them into limit figures, which can be found in the [figures](../figures/) folder.





