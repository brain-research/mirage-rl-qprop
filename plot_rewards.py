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
    descs = [fname.split('/')[2] for fname in fnames]
    descs = sorted(descs)
    algos = [algo.split('-')[0] for algo in descs]
    env = descs[0].split('-')[1]
    descs = [x.replace('-{}'.format(env), '') for x in descs]
    descs_d = dict((algo, color[i]) for i, algo in enumerate(set(algos)))
    print("\nDescriptions:")
    print("\n".join(descs))
    print("\nAlgorithms Used:")
    print("\n".join(set(algos)))

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    ax.set_title('Average reward ({})'.format(env))
    prev_algo = ""
    for i, fname in enumerate(fnames):
        fh = open(fname)
        fields = fh.readline()
        fh.close()
        data = np.genfromtxt(fname, delimiter=',', skip_header=2, skip_footer=0, names=fields.split(','))
        eps = data['Iteration'][:eps_limit] * 5000 / 1000000
        rews = data['AverageReturn'][:eps_limit]
        algo = fname.split('/')[2].split('-')[0]
        if algo != prev_algo:
            plt.plot(eps, savgol_filter(rews, 101, 5), color=descs_d[algo], label="{}".format(descs[i].split('-')[0]))
        else:
            plt.plot(eps, savgol_filter(rews, 101, 5), color=descs_d[algo])
        plt.plot(eps, rews, color=descs_d[algo], alpha=0.1)
        prev_algo = algo

    plt.legend(loc='lower center', ncol=3)
    plt.xlabel('steps in millions (batch_size=5000)')
    plt.ylabel('average reward')
    # axes = plt.gca()
    # axes.set_ylim([-700, 5000])
    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    plot_filename = "plots/{}_{}.png".format("-".join(descs), eps_limit)[:256]
    plt.savefig(plot_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot average rewards of experiments.")
    parser.add_argument('--files', help="Pass in regex-style for filenames; split regexes by comma to capture different sets filenames", required=True)
    args = parser.parse_args()

    fnames = []
    files_list = args.files.split(',')
    for files in files_list:
        fnames.extend(glob.glob(files))

    fnames = [fname for fname in fnames if fname.split('/')[2] != 'test']
    print("\n".join(fnames))

    main(fnames)
