# Code

## Yellin method preparation

Run [script_mc_vols.py](script_mc_vols.py) with syntax `script_mc_vols.py $i` with `$i` running over a large set of integers (default: 200). This integer runs over the elements of `list_mu`, and is the number of expected counts inside the unit cuboid. This will generate a large number (`N_MC`) of Monte Carlo volumes for each `mu` in the specified output directory. They are written to a directory called `dir_ceph`, and are to be used for the later Yellin analysis. (Only needs to be done once.)

## Yellin/Poisson analysis

1. Run [script_yellin_proj_fid_mock.py](script_yellin_proj_fid_mock.py) with syntax `yellin_proj_fid_mock.py $jmock $im` to project photon events from mock data onto the 4D unit cuboid. For the mocks, `$jmock$` denotes an integer labeling the mock data set. The integer `$im` runs over the different axion masses. For the observed data, the syntax is `yellin_proj_fid_data.py $im`.

2. Run [script_yellin_vols.py](script_yellin_vols.py) with syntax `yellin_vols.py $jmock $im $k`. 



