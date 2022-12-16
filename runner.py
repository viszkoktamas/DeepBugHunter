#
# This file just serves to give a better intuition of how we conducted batch experiments
# It is not strictly a part of the DBH infrastructure, just an automation layer one level above it
#
import os
import copy
import pickle
from pathlib import Path

import dbh
import dbh_util as util

shared = {
    'csv': 'dataset.csv',
    'label': 'label',
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
    # ['sdnnc', '--layers 5 --neurons 1024 --batch 16 --epochs 10 --lr 0.1'],
    # ['forest', '--max-depth 10 --criterion entropy --n-estimators 5'],
]


def main():
    for data_step in data_steps:
        params = copy.deepcopy(shared)
        params = {**params, **data_step, 'strategy': basic_strategy}
        res = dbh.main(params)
        return res[0][7]


def full_train(batch_pretrain, batch_train):
    pre_train_conf = f'--layers 5 --neurons 1024 --batch {batch_pretrain} --epochs 10 --lr 0.1'
    train_conf_with = f'--layers 5 --neurons 1024 --batch {batch_train} --epochs 10 --lr 0.1 --pretrain layers-5_neurons-1024_batch-{batch_pretrain}_epochs-10_lr-0.1_beta-0.0'
    train_conf_without = f'--layers 5 --neurons 1024 --batch {batch_train} --epochs 10 --lr 0.1'

    basic_strategy[0][1] = pre_train_conf
    shared["csv"] = "dataset_warn_full.csv"
    main()
    basic_strategy[0][1] = train_conf_with
    shared["csv"] = "dataset_vuln_full.csv"
    pretrained_f_measure = main()

    basic_strategy[0][1] = train_conf_without
    shared["csv"] = "dataset_vuln_full.csv"
    not_pretrained_f_measure = main()

    return pretrained_f_measure, not_pretrained_f_measure, train_conf_with, train_conf_without


def only_pretrain(csv="dataset_warn_full.csv", layers=5, neurons=1024, batch=512, epochs=10, lr=0.1):
    pre_train_conf = f'--layers {layers} --neurons {neurons} --batch {batch} --epochs {epochs} --lr {lr}'

    basic_strategy[0][1] = pre_train_conf
    shared["csv"] = csv
    return main(), pre_train_conf


def full_train_test():
    for batch_pretrain, batch_train in [(256, 256), (256, 128), (256, 64), (256, 32), (256, 16)]:
        pretrained_f_measure, not_pretrained_f_measure, train_conf_with, train_conf_without = full_train(batch_pretrain, batch_train)
        with open("results.csv", 'a') as f:
            f.write(f"{pretrained_f_measure},{not_pretrained_f_measure},{train_conf_with},{train_conf_without}\n")


def check_path(path):
    path.parent.mkdir(exist_ok=True)
    path.touch(exist_ok=True)
    return path


def pre_train_test():
    layers = [3, 4, 5, 6, 7]
    neurons = [256, 512, 1024, 2048]
    batches = [128, 256, 512, 1024]
    epochs = [5]
    lrs = [.1, .01, .001, .0001]
    csvs = ["dataset_warn_full.csv", "dataset_warn_without_minor.csv"]

    for l in layers:
        for n in neurons:
            for b in batches:
                for e in epochs:
                    for lr in lrs:
                        for csv in csvs:
                            pickle_file = Path("pickles") / Path("_".join(map(lambda x: str(x), [csv, l, n, b, e, lr])))
                            if pickle_file.exists():
                                continue

                            res, conf = only_pretrain(csv, l, n, b, e, lr)
                            with open("results_pretrain.csv", 'a') as f:
                                f.write(f"{res},{conf} {csv}\n")

                            with open(check_path(pickle_file), "wb") as f:
                                pickle.dump(pickle_file, f)


if __name__ == '__main__':
    pre_train_test()
