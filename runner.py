#
# This file just serves to give a better intuition of how we conducted batch experiments
# It is not strictly a part of the DBH infrastructure, just an automation layer one level above it
#
import os
import copy

import dbh

shared = {
    'csv': 'dataset.csv',
    'label': 'BUG',
    'clean': False,
    'seed': 1337,
    'output': os.path.abspath('output'),
    'device': '/device:CPU:0',
    'log_device': False,
    'calc_completeness': True
}

data_steps = [
    {
        'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
        'resample': 'none',
        'resample_amount': 0
    },
]

basic_strategy = [
    ['keras', '--layers 5 --neurons 1024 --batch 16 --epochs 10 --lr 0.1'],
    ['sdnnc', '--layers 5 --neurons 1024 --batch 16 --epochs 10 --lr 0.1'],
    ['forest', '--max-depth 10 --criterion entropy --n-estimators 5'],
]


def main():
    for data_step in data_steps:
        params = copy.deepcopy(shared)
        params = {**params, **data_step, 'strategy': basic_strategy}
        dbh.main(params)
    

if __name__ == '__main__':
    main()
