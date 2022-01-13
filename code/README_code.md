# Code

## Yellin/Poisson analysis

### Yellin method preparation

0. Run [script_00_yellin_mc_vols.py](script_00_yellin_mc_vols.py) with syntax `script_00_yellin_mc_vols.py $i` with `$i` running over a large set of integers (default: 200). This integer runs over the elements of `list_mu`, and is the number of expected counts inside the unit cuboid. This will generate a large number (`N_MC`) of Monte Carlo volumes for each `mu` in the specified output directory. They are written to a directory called `dir_ceph`, and are to be used for the later Yellin analysis. (Only needs to be done once but takes a long time.)

1. Run [script_01_yellin_cmax_bar.py](script_01_yellin_cmax_bar.py) with syntax `script_01_yellin_cmax_bar.py $i` with `$i` running over the same set of integers. This will compute C_max_bar as a function of mu based on the Monte Carlo, and save the result in a csv file in the `dir_mc` folder. (This only needs to be done once.)

### Yellin/Poisson scripts on mocks (with fiducial solar position)

2. Run [script_02_yellin_proj_fid_mock.py](script_02_yellin_proj_fid_mock.py) with syntax `script_02_yellin_proj_fid_mock.py $jmock $im` to project photon events from mock data onto the 4D unit cuboid. For the mocks, `$jmock$` denotes an integer labeling the mock data set. The integer `$im` runs over the different axion masses. 

3. Run [script_03_yellin_vols_mock.py](script_03_yellin_vols_mock.py) with syntax `script_03_yellin_vols_mock.py $jmock $im $k` to compute the Yellin volumes on the projection files from the previous step. 

4. Run [script_04_yellin_cmax_mock.py](script_04_yellin_cmax_mock.py) with syntax `script_04_yellin_cmax_mock.py $jmock $im $k` to compute C_max values based on the CDFs from the Monte Carlo run (step 0) and the computed volumes (step 3) on the mock data.

### Yellin/Poisson scripts on data


