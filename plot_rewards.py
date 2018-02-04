import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches

import argparse
import glob

import numpy as np
import seaborn as sns
import scipy
from scipy.signal import savgol_filter

def main(args):
    # Plot average rewards
    savgol_window = 21
    experiments_list = args.split('|')
    fig = plt.figure(figsize=(20, 6))
    color_list = []
    color_list.append((147, 79, 168)) # Darker purple (bottom)
    color_list.append((22, 170, 71)) # Darker green (top)
    color_list.append((28, 124, 188)) # Darker blue (middle)
    color_list = np.array(color_list) / 255.0
    eps_list = [100, 1000, 1000]

    map_algos_colors = dict()
    eps_limit = dict()
    env_names = dict()
    for i, key in enumerate(['trpo', 'qpropconserv', 'qpropconserveta']):
      map_algos_colors[key] = color_list[i]

    env_names['cartpole'] = 'CartPole'
    eps_limit['cartpole'] = eps_list[0]
    env_names['halfcheetah'] = 'HalfCheetah'
    eps_limit['halfcheetah'] = eps_list[1]
    env_names['humanoid'] = 'Humanoid'
    eps_limit['humanoid'] = eps_list[2]

    for idx, experiment_files_list in enumerate(experiments_list):
        files_list = experiment_files_list.split(',')
        fnames = []
        for files in files_list:
            fnames.extend(glob.glob(files))

        fnames = [fname for fname in fnames if fname.split('/')[2] != 'test']
        print("\n".join(fnames))

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

        algos_steps = dict()
        algos_rews = dict()
        ax = plt.subplot(131 + idx)
        ax.set_title(env_names[env], fontsize=16)

        for i, fname in enumerate(fnames):
            fh = open(fname)
            fields = fh.readline()
            fh.close()
            data = np.genfromtxt(fname, delimiter=',', skip_header=2, skip_footer=0, names=fields.split(','))
            env = fname.split('-')[1]
            algo = fname.split('/')[2].split('-')[0]
            limit = eps_limit[env] # Assumes fname is data/local/NAME-stuff-stuff...
            steps = data['Iteration'][:limit] * 5000
            rews = data['AverageReturn'][:limit]

            if algo not in algos_steps:
                algos_steps[algo] = steps
            if algo not in algos_rews:
                algos_rews[algo] = [savgol_filter(rews, savgol_window, 5)]
            else:
                algos_rews[algo].append(rews)

        for algo in algos_rews.keys():
            algos_rews[algo] = np.stack(algos_rews[algo])

        for algo in reversed(sorted(algos_rews.keys())):
            rews_z1 = savgol_filter(algos_rews[algo].mean(0) + algos_rews[algo].std(0), savgol_window, 5)
            rews_z_1 = savgol_filter(algos_rews[algo].mean(0) - algos_rews[algo].std(0), savgol_window, 5)
            rews_max = savgol_filter(algos_rews[algo].max(0), savgol_window, 5)
            rews_min = savgol_filter(algos_rews[algo].min(0), savgol_window, 5)
            rews_mean = savgol_filter(algos_rews[algo].mean(0), savgol_window, 5)
            plt.plot(algos_steps[algo]/1000000, rews_mean, color=map_algos_colors[algo], alpha=1.0, label=algo, linewidth=2.10)
            plt.fill_between(algos_steps[algo]/1000000, rews_mean, np.where(rews_z1 > rews_max, rews_max, rews_z1), color=map_algos_colors[algo], alpha=0.4)
            plt.fill_between(algos_steps[algo]/1000000, np.where(rews_z_1 < rews_min, rews_min, rews_z_1), rews_mean, color=map_algos_colors[algo], alpha=0.4)

            # plt.xlabel('Steps (in millions) (batch_size=5000)')
            # plt.ylabel('Average Reward')

    green = mpatches.Patch(color=color_list[1], label='Original QProp (biased)')
    blue = mpatches.Patch(color=color_list[2], label='Corrected QProp (unbiased)')
    purple = mpatches.Patch(color=color_list[0], label='TRPO (unbiased)')
    leg = fig.legend(handles=[green, blue, purple], loc='lower center', ncol=3, prop={'size': 14})


    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    plot_filename = "plots/{}-{}.png".format("-".join(['trpo', 'qpropc', 'qpropceta']), 'all')[:256]
    plt.savefig(plot_filename)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Plot average rewards of experiments.")
    # parser.add_argument('--files', help="Pass in regex-style for filenames; split regexes by comma to capture different sets filenames; split regexes by | to capture different sets of experiments", required=True)
    # args = parser.parse_args()
    # Command:
    # python plot_rewards.py --files=data/local/qpropconserv*cartpole*/*/progress.csv,data/local/trpo*cartpole*/*/progress.csv|data/local/qpropconserv*halfcheetah*/*/progress.csv,data/local/trpo*halfcheetah*/*/progress.csv|data/local/qpropconserv*humanoid*/*/progress.csv,data/local/trpo*humanoid*/*/progress.csv

    args = 'data/local/qpropconserv*cartpole*/*/progress.csv,data/local/trpo*cartpole*/*/progress.csv|data/local/qpropconserv*halfcheetah*/*/progress.csv,data/local/trpo*halfcheetah*/*/progress.csv|data/local/qpropconserv*humanoid*/*/progress.csv,data/local/trpo*humanoid*/*/progress.csv'
    # main(args.files)
    main(args)
