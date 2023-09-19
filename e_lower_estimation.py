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
parser.add_argument('--e_max', type=float, required=True)
parser.add_argument('--e_min', type=float, required=True)
parser.add_argument('--e_lower', type=float, required=False)
parser.add_argument('--e_charge', type=float, required=False)
parser.add_argument('--resolution', type=float, nargs='?', default=0.1, required=False)
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
E_lower \in [e_min, e_charge * 0.95]
E_charge = 7/8 * e_max
"""
resolution = args.resolution
e_max = args.e_max
e_min = args.e_min
e_charge = 7/8 * e_max
e_lowers = np.arange(e_min, e_charge * 0.95, resolution)[1:]

"""
Loop
"""
for e_lower in e_lowers:
    subprocess.run(["python", "persistification_learning.py", 
                    "--e_min", str(e_min),
                    "--e_max", str(e_max), 
                    "--e_lower", str(e_lower), 
                    "--e_charge", str(e_charge),
                    "--out_file", out_file]
    )