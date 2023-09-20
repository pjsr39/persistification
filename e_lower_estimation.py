# run file persistifcation learning.py
import subprocess
import numpy as np
import argparse
import calendar
import time
import os

"""
Args
"""
parser = argparse.ArgumentParser()
parser.add_argument('--e_min', type=float, required=True)
parser.add_argument('--e_init', type=float, required=True)
parser.add_argument('--e_charge', type=float, required=True)
parser.add_argument('--resolution', type=float, nargs='?', default=0.1, required=False)
parser.add_argument('--animation', action='store_true')
args = parser.parse_args()

"""
Dir timestamp
"""
current_GMT = time.gmtime()
ts = str(calendar.timegm(current_GMT))
wd = os.path.join(os.getcwd(),'e_lower_estimation_runs')
if not os.path.exists(wd):
    os.mkdir(wd)
folder_name = 'e_lower_estimation_run_' + ts
os.mkdir(os.path.join(wd, folder_name))
folder_dir = os.path.join(wd, folder_name)
out_file = os.path.join(folder_dir, "distances.txt")
f = open(out_file, "x")

"""
Params
note: E_lower \in [e_min, e_charge]
"""
resolution = args.resolution
e_init = args.e_init
e_min = args.e_min
e_charge = args.e_charge
e_lowers = np.arange(e_min, e_charge, resolution)[1:]

"""
Loop
"""
cmd = ["python3", "persistification_learning.py", 
                    "--e_min", str(e_min),
                    "--e_init", str(e_init), 
                    "--e_lower", "", 
                    "--e_charge", str(e_charge),
                    "--out_file", out_file
                    ]
if args.animation: cmd.append("--animation")

for e_lower in e_lowers:
    cmd[7] = str(e_lower)
    subprocess.run(cmd)