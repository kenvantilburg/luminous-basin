# Code

## Yellin method preparation

First, run [script_mc_vols.py](code/script_mc_vols.py) with syntax `script_mc_vols.py $i` with `$i` running over a large set of integers (default: 200). This integer runs over the elements of `list_mu`, and is the number of expected counts inside the unit cuboid. This will generate a large number (`N_MC`) of Monte Carlo volumes for each `mu`.