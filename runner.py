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
    'csv': 'dataset_vuln_full.csv',
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


def full_train(csv, l, n, b, e, lr, r, ra):
    train_conf = f'--layers {l} --neurons {n} --batch {b} --epochs {e} --lr {lr}'
    pre_train_id = f'layers-{l}_neurons-{n}_batch-{b}_epochs-{e}_lr-{lr}_beta-0.0_{r}_{ra}_1337_{csv[:-4]}'
    train_conf_with_pretrain = f'{train_conf} --pretrain {pre_train_id}'

    data_step = {
        'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
        'resample': r,
        'resample_amount': ra
    }

    basic_strategy[0][1] = f'{train_conf} --save'
    data_steps[0] = data_step
    shared["csv"] = csv
    pretrain_f_measure = main()
    basic_strategy[0][1] = train_conf_with_pretrain
    data_steps[0] = data_step
    shared["csv"] = "dataset_vuln_full.csv"
    pretrained_f_measure = main()

    basic_strategy[0][1] = train_conf
    data_steps[0] = data_step
    shared["csv"] = "dataset_vuln_full.csv"
    not_pretrained_f_measure = main()

    return pretrain_f_measure, pretrained_f_measure, not_pretrained_f_measure, train_conf_with_pretrain


def only_pretrain(csv="dataset_warn_full.csv", layers=5, neurons=1024, batch=512, epochs=10, lr=0.1, resample='none', r_amount=0):
    pre_train_conf = f'--layers {layers} --neurons {neurons} --batch {batch} --epochs {epochs} --lr {lr}'
    data_step = {
        'preprocess': [['features', 'standardize'], ['labels', 'binarize']],
        'resample': resample,
        'resample_amount': r_amount
    }

    basic_strategy[0][1] = pre_train_conf
    data_steps[0] = data_step
    shared["csv"] = csv
    return main(), f'{pre_train_conf}_{resample}_{r_amount}'


def do_train(csv, l, n, b, e, lr, r, ra):
    pretrain_f1, pretrained_f1, not_pretrained_f1, train_conf = full_train(csv, l, n, b, e, lr, r, ra)
    with open("results2.csv", 'a') as f:
        f.write(f"{pretrain_f1},{pretrained_f1},{not_pretrained_f1},{train_conf}\n")


def full_train_test():
    layers = [7, 5, 3]
    neurons = [2048, 1024, 512]
    batches = [512, 256, 128]
    epochs = [5, 10, 15, 20]
    lrs = [.0001]
    resample = ['none', 'up', 'down']
    r_amount = [0, 50, 100]
    csvs = ["dataset_warn_full.csv", "dataset_warn_without_minor.csv", "dataset_warn_without_minor_and_major.csv"]

    for l in layers:
        for n in neurons:
            for b in batches:
                for e in epochs:
                    for lr in lrs:
                        for r in resample:
                            for ra in r_amount:
                                for csv in csvs:
                                    if (r == 'none' and ra != 0) or (r != 'none' and ra == 0):
                                        continue

                                    pickle_file = Path("pickles") / Path("_".join(map(lambda x: str(x), [csv, l, n, b, e, lr, r, ra])))
                                    if pickle_file.exists():
                                        continue

                                    do_train(csv, l, n, b, e, lr, r, ra)
                                    touch_path(pickle_file)


def touch_path(path):
    path.parent.mkdir(exist_ok=True)
    path.touch(exist_ok=True)
    return path


def pre_train_test():
    layers = [3, 5, 7]
    neurons = [512, 1024, 2048]
    batches = [128, 256, 512]
    epochs = [10, 15, 20]
    lrs = [.0001]
    resample = ['none']
    r_amount = [0]
    csvs = ["dataset_warn_full.csv"]

    for l in layers:
        for n in neurons:
            for b in batches:
                for e in epochs:
                    for lr in lrs:
                        for r in resample:
                            for ra in r_amount:
                                for csv in csvs:
                                    pickle_file = Path("pickles") / Path("_".join(map(lambda x: str(x), [csv, l, n, b, e, lr, r, ra])))
                                    if pickle_file.exists():
                                        continue

                                    res, conf = only_pretrain(csv, l, n, b, e, lr, r, ra)
                                    with open("results_pretrain.csv", 'a') as f:
                                        f.write(f"{res},{conf} {csv}\n")

                                    with open(touch_path(pickle_file), "wb") as f:
                                        pickle.dump(pickle_file, f)


if __name__ == '__main__':
    full_train_test()
    # pre_train_test()
