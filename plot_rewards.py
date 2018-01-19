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

def main(fnames):
    # Plot average rewards

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    descs = [fname.split('/')[2] for fname in fnames]

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    ax.set_title('Average reward')
    for i, fname in enumerate(fnames):
        fh = open(fname)
        fields = fh.readline()
        fh.close()

        data = np.genfromtxt(fname, delimiter=',', skip_header=0, skip_footer=0, names=fields.split(','))
        eps = data['Iteration'][:4000]
        rews = data['AverageReturn'][:4000]
        plt.plot(eps, savgol_filter(rews, 101, 5), color[i], label="{}".format(descs[i]))
        plt.plot(eps, rews, color[i], alpha=0.175)

    plt.legend(loc='lower center', ncol=3)
    # axes = plt.gca()
    # axes.set_ylim([-700, 5000])
    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    plot_filename = "plots/{}.png".format("-".join(descs))
    plt.savefig(plot_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot average rewards of experiments.")
    parser.add_argument('--files', help="Pass in regex-style for filenames", required=True)
    args = parser.parse_args()

    fnames = glob.glob(args.files)
    print("\n".join(fnames))
    main(fnames)
