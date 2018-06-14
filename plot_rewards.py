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

from matplotlib import rc
rc('text', usetex=True)

import seaborn as sns
color_list = sns.color_palette("muted")
sns.palplot(color_list)


def main(args):
    # Plot average rewards
    experiments_list = args.files.split('|')
    fig = plt.figure(figsize=(20, 6))
    eps_list = [24, 1000, 1000]

    map_algos_colors = dict()
    eps_limit = dict()
    env_names = dict()
    for i, key in enumerate(['trpo', 'qpropconserv', 'qpropconserveta']):
      map_algos_colors[key] = color_list[i]

    algo_names = {'trpo': 'TRPO', 'qpropconserv': 'QProp (biased)', 'qpropconserveta': 'QProp (unbiased)'}

    env_names['cartpole'] = 'CartPole-v0'
    eps_limit['cartpole'] = eps_list[0]
    env_names['halfcheetah'] = 'HalfCheetah-v1'
    eps_limit['halfcheetah'] = eps_list[1]
    env_names['humanoid'] = 'Humanoid-v1'
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
        ax.set_title(env_names[env], fontsize=18)
        ax.grid(alpha=0.5)

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

            if env == 'cartpole':
              savgol_window = 5
              poly_order = 3
            else:
              savgol_window = 25
              poly_order = 5

            if algo not in algos_steps:
                algos_steps[algo] = steps
            if algo not in algos_rews:
                algos_rews[algo] = [savgol_filter(rews, savgol_window, poly_order)]
            else:
                algos_rews[algo].append(rews)

        for algo in algos_rews.keys():
            algos_rews[algo] = np.stack(algos_rews[algo])

        for algo in reversed(sorted(algos_rews.keys())):
            rews_z1 = savgol_filter(algos_rews[algo].mean(0) + algos_rews[algo].std(0), savgol_window, poly_order)
            rews_z_1 = savgol_filter(algos_rews[algo].mean(0) - algos_rews[algo].std(0), savgol_window, poly_order)
            rews_max = savgol_filter(algos_rews[algo].max(0), savgol_window, poly_order)
            rews_min = savgol_filter(algos_rews[algo].min(0), savgol_window, poly_order)
            rews_mean = savgol_filter(algos_rews[algo].mean(0), savgol_window, poly_order)
            plt.plot(algos_steps[algo]/1000, rews_mean, color=map_algos_colors[algo], alpha=1.0, label=algo_names[algo])
            plt.fill_between(algos_steps[algo]/1000, rews_mean, np.where(rews_z1 > rews_max, rews_max, rews_z1), color=map_algos_colors[algo], alpha=0.4)
            plt.fill_between(algos_steps[algo]/1000, np.where(rews_z_1 < rews_min, rews_min, rews_z_1), rews_mean, color=map_algos_colors[algo], alpha=0.4)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tick_params(axis='both', which='minor', labelsize=12)

            plt.xlabel('Steps (thousands)', fontsize=14)
            plt.ylabel('Average Reward', fontsize=14)
        if args.mini:
            plt.legend(loc='lower right', prop={'size': 15})


    if not args.mini:
        green = mpatches.Patch(color=map_algos_colors['qpropconserv'], label='QProp (biased)')
        blue = mpatches.Patch(color=map_algos_colors['qpropconserveta'], label='QProp (unbiased)')
        purple = mpatches.Patch(color=map_algos_colors['trpo'], label='TRPO')
        leg = fig.legend(handles=[green, blue, purple], loc='lower center', ncol=3, prop={'size': 16})

        # Move legend down.
        bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
        bb.y0 += -0.10
        leg.set_bbox_to_anchor(bb, transform = ax.transAxes)


    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    if args.mini:
        suffix = 'all-mini'
    else:
        suffix = 'all'

    plot_filename = "plots/{}-{}.pdf".format("-".join(['trpo', 'qpropc', 'qpropceta']), suffix)[:256]
    plt.savefig(plot_filename, bbox_inches='tight', dpi=200, format='pdf')

if __name__ == '__main__':
    default_fnames = 'data/local/qpropconserv*cartpole*/*/progress.csv,data/local/trpo*cartpole*/*/progress.csv|data/local/qpropconserv*halfcheetah*/*/progress.csv,data/local/trpo*halfcheetah*/*/progress.csv|data/local/qpropconserv*humanoid*/*/progress.csv,data/local/trpo*humanoid*/*/progress.csv'

    parser = argparse.ArgumentParser(description="Plot average rewards of experiments.")
    parser.add_argument('--files', type=str, default=default_fnames, metavar='S', help="Pass in regex-style for filenames; split regexes by comma to capture different sets filenames; split regexes by | to capture different sets of experiments", required=True)
    parser.add_argument('--mini', action='store_true', help='generate the individual mini plots that can be cropped with legends')
    args = parser.parse_args()

    main(args)
