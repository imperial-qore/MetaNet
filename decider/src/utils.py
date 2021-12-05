import os
import pickle
import numpy as np

DATASET_PATH = './decider/src/datasets/'
FN_PATH = './functions/'

def get_response_times(apps, choices):
    fname = DATASET_PATH + 'random_stats.pk'
    with open(fname, 'rb') as handle:
        stats = pickle.load(handle)
    rt_dict = dict(zip(apps, [dict(zip(choices, [[] for _ in range(3)])) for _ in range(len(apps))]))
    for r in stats.alltaskinfo:
        rt_dict[r['application']][r['choice']].append(r['destroyAt'] - r['createAt'])
    return rt_dict

def get_dicts():
    apps = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]
    choices = ['layer', 'semantic', 'compression']
    rt_dict = get_response_times(apps, choices); sla_dict = dict(zip(apps, [0 for _ in range(len(apps))]))
    for app in apps:
        lsts = []
        for choice in choices:
            lst = rt_dict[app][choice]; lsts += lst
            rt_dict[app][choice] = (np.mean(lst), np.std(lst)) if lst else (0, 0)
        sla_dict[app] = np.percentile(lsts, 55)
    return rt_dict, sla_dict

def get_accuracies():
    apps = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]
    accs = dict(zip(apps, [0.7, 0.8, 0.85, 0.92, 0.97, 0.89, 0.76]))
    choices = ['layer', 'semantic', 'compression']
    choice_multiplier = dict(zip(choices, [1, 0.8, 0.6]))
    a_dict = dict(zip(apps, [{} for _ in range(len(apps))]))
    for app in apps:
        for choice in choices:
            a_dict[app][choice] = accs[app] * choice_multiplier[choice]
    return a_dict