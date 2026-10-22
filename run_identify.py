"""Run a HySI-SAT identification without the GUI (e.g. as a DelftBlue batch job).

Run it from the folder where results should go (e.g. a /scratch run folder); input
paths are relative to that folder. The folder is prepared automatically: a fresh
DataOld.db is copied from the repo and Data/IdentifiedDatasets/ is created.

Usage:
    python run_identify.py --input Data/inputHVAC_first_1000.txt \
        --output Data/outputHVAC_first_1000.txt --nu 1 --ny 1 --dw 15 --delta 0.8 --blocks 2
"""
import argparse
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import numpy as np

import identify

REPO = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="input file (one comma-separated row per input)")
p.add_argument("--output", required=True, help="output file (one comma-separated row per output)")
p.add_argument("--nu", type=int, default=1, help="input lags")
p.add_argument("--ny", type=int, default=1, help="output lags")
p.add_argument("--dw", type=int, default=15, help="minimum dwell time")
p.add_argument("--delta", type=float, default=0.1, help="noise bound")
p.add_argument("--blocks", type=int, default=1, help="number of data blocks")
p.add_argument("--no-split", action="store_true", help="do not split large datasets")
p.add_argument("--pwarx", action="store_true", help="identify a PWARX instead of a SARX model")
p.add_argument("--Ls", type=int, default=1)
p.add_argument("--seed", type=int, default=7)
args = p.parse_args()

if not os.path.exists("DataOld.db"):
    shutil.copy(os.path.join(REPO, "pythonfiles", "DataOld.db"), "DataOld.db")
os.makedirs("Data/IdentifiedDatasets", exist_ok=True)

n, sigma, error, mse, t, Fmodels = identify.main(
    None, False, inputfile=args.input, outputfile=args.output, nu=args.nu, ny=args.ny,
    dw=args.dw, delta=args.delta, blocks=args.blocks, split=int(not args.no_split),
    pwarx=int(args.pwarx), Ls=args.Ls, seed=args.seed, modelgeneration=2)

print("Number of models:", len(Fmodels))
print("Data fit error:", float(np.sum(error)))
print("Switching sequence:", list(map(int, sigma)))
print("Time: %.1f s" % t)
