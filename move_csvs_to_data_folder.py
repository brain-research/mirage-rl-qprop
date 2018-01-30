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
        directory = fname.split("-stdout")[0]
        if not os.path.exists(directory):
            os.makedirs('data/local/{}'.format(directory))
            os.makedirs('data/local/{}/{}'.format(directory, 'placeholder'))

        os.rename(fname, 'data/local/{}/placeholder/{}'.format(directory, 'progress.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot average rewards of experiments.")
    parser.add_argument('--files', help="Pass in regex-style for filenames", required=True)
    args = parser.parse_args()

    fnames = glob.glob(args.files)
    # fnames = [fname for fname in fnames if fname.split('/')[2] != 'test']
    print("\n".join(fnames))

    main(fnames)

