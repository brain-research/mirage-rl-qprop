import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import argparse
import glob

import numpy as np
import seaborn as sns
import scipy
from scipy.signal import savgol_filter

# Convert the nasty stdout files to CSV files
def main(fnames):
    for i, fname in enumerate(fnames):
        with open(fname) as f:
            content = f.readlines()

        rets, its = [], []
        for line in content:
            if 'AverageReturn' in line:
                rets.append(float(line.split()[-1]))
            if 'Iteration' in line:
                its.append(int(line.split()[-1]))

        with open('{}.csv'.format(fname), 'w+') as f:
            f.write('Iteration,AverageReturn\n')
            for it, ret in zip(its, rets):
                f.write('{},{}\n'.format(it, ret))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot average rewards of experiments.")
    parser.add_argument('--files', help="Pass in regex-style for filenames", required=True)
    args = parser.parse_args()

    fnames = glob.glob(args.files)
    # fnames = [fname for fname in fnames if fname.split('/')[2] != 'test']
    print("\n".join(fnames))

    main(fnames)


