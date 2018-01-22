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

    eps_limit = 2000
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fnames = sorted(fnames)
    descs = [fname.split('/')[2] for fname in fnames]
    algos = [algo.split('-')[0] for algo in descs]
    env = descs[0].split('-')[1]
    descs = [x.replace('-{}'.format(env), '') for x in descs]
    descs_d = dict((algo, color[i]) for i, algo in enumerate(set(algos)))
    print("\n".join(descs))
    print("\n".join(algos))

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    ax.set_title('Average reward ({})'.format(env))
    for i, fname in enumerate(fnames):
        fh = open(fname)
        fields = fh.readline()
        fh.close()

        data = np.genfromtxt(fname, delimiter=',', skip_header=0, skip_footer=0, names=fields.split(','))
        eps = data['Iteration'][:eps_limit]
        rews = data['AverageReturn'][:eps_limit]
        algo = fname.split('/')[2].split('-')[0]
        plt.plot(eps, savgol_filter(rews, 101, 5), descs_d[algo], label="{}".format(descs[i]))
        plt.plot(eps, rews, descs_d[algo], alpha=0.175)

    plt.legend(loc='lower center', ncol=3)
    # axes = plt.gca()
    # axes.set_ylim([-700, 5000])
    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    plot_filename = "plots/{}_{}.png".format("-".join(descs), eps_limit)
    plt.savefig(plot_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot average rewards of experiments.")
    parser.add_argument('--files', help="Pass in regex-style for filenames", required=True)
    args = parser.parse_args()

    fnames = glob.glob(args.files)
    fnames = [fname for fname in fnames if fname.split('/')[2] != 'test']
    print("\n".join(fnames))

    main(fnames)
