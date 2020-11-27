import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

def df_list(exps,alg_type):
    """
    Inputs:
        exps     -- list of csv file paths to turn into list of dataframes
        alg_type -- string indicating which algorithm was used
    """
    df = []
    # iterate over the files
    for e in exps:
        this_df = pd.read_csv(e)
        # assign new column for algorithm if needed
        if 'algorithm' not in this_df.columns:
            this_df['algorithm'] = alg_type
            print(f'added {alg_type} to algorithm column')
        # rename ' reward' to 'reward'
        if ' reward' in this_df.columns:
            this_df.columns = ['reward' if x==' reward' else x for x in this_df.columns]
        # append to list
        df.append(this_df)
        # rename eprewmean
        if 'eprewmean' in this_df.columns:
            this_df.columns = ['reward_tr' if x=='eprewmean' else x for x in this_df.columns]
        # assign new column if iteration is not there
        if 'iteration' not in this_df.columns:
            num_rows = this_df.count()['reward_tr']
            its = list(range(1,num_rows+1))
            this_df['iteration'] = its
            print(f'added its for {alg_type}')
    return df

# get all asgf, dgs, and es csv files in directory
asgf = df_list(glob.glob('./asgf*.csv'), 'asgf')
dgs = df_list(glob.glob('./dgs*.csv'), 'dgs')
es_1000 = df_list(glob.glob('./es_1000*.csv'), 'es_1000')
es_1500 = df_list(glob.glob('./es_1500*.csv'), 'es_1500')
es_2000 = df_list(glob.glob('./es_2000*.csv'), 'es_2000')
df = pd.concat(asgf + dgs + es_1000 + es_1500 + es_2000, axis=0)
print(df.columns)

# add the 'sample' column (I am ashamed of this code but it works)
df['sample'] = df['iteration']
df['sample'] *= [5*113 if alg=='asgf' else 1 for alg in df['algorithm']]
df['sample'] *= [7*113 if alg=='dgs' else 1 for alg in df['algorithm']]
df['sample'] *= [1000 if alg=='es_1000' else 1 for alg in df['algorithm']]
df['sample'] *= [1500 if alg=='es_1500' else 1 for alg in df['algorithm']]
df['sample'] *= [2000 if alg=='es_2000' else 1 for alg in df['algorithm']]

# plot reward vs iterations
env_name = os.getcwd().split('/')[-1]
sns.set_palette(sns.color_palette())
with sns.axes_style('whitegrid'):
    fig = plt.figure(figsize=(6,4))
    ax = sns.lineplot(x='iteration', y='reward', hue='algorithm',\
        data=df, linewidth=3, alpha=.75)
    ax.set_title(env_name + ' -- training reward')
    plt.xlabel('Policy updates')
    plt.xlim([0,100])
    plt.ylabel('Average reward')
    legend = ax.legend(loc='lower right')
    for line in legend.get_lines():
        line.set_linewidth(3)
    plt.tight_layout()
    plt.savefig('./reward_iteration_{:s}.pdf'.format(env_name), dpi=300, format='pdf')
    plt.show()

# plot reward vs samples
with sns.axes_style('whitegrid'):
    fig = plt.figure(figsize=(6,4))
    ax = sns.lineplot(x='sample', y='reward', hue='algorithm',\
        data=df, linewidth=3, alpha=.75)
    ax.set_title(env_name + ' -- training reward')
    plt.xlabel('Episodes')
    plt.xlim([0,100000])
    plt.ylabel('Average reward')
    legend = ax.legend(loc='lower right')
    for line in legend.get_lines():
        line.set_linewidth(3)
    plt.tight_layout()
    plt.savefig('./reward_sample_{:s}.pdf'.format(env_name), dpi=300, format='pdf')
    plt.show()




