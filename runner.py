#
# This file just serves to give a better intuition of how we conducted batch experiments
# It is not strictly a part of the DBH infrastructure, just an automation layer one level above it
#
import re
import os
import copy
import pickle
from pathlib import Path
from bayes_opt import BayesianOptimization, UtilityFunction

import dbh

shared = {
    'csv': 'dataset_vuln_full.csv',
    'label': 'bug',
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
    ['keras', '--layers 5 --neurons 1024 --batch 16 --epochs 10 --lr 0.1 --save'],
    # ['sdnnc', '--layers 5 --neurons 1024 --batch 16 --epochs 10 --lr 0.1'],
    # ['forest', '--max-depth 10 --criterion entropy --n-estimators 5'],
]


def main():
    for data_step in data_steps:
        params = copy.deepcopy(shared)
        params = {**params, **data_step, 'strategy': basic_strategy}
        res = dbh.main(params)
        return res[0][7]


def only_train(csv="dataset_vuln_full.csv", l=5, n=1024, b=512, e=10, lr=0.1, r='none', ra=0, save=False,
               pretrain_conf=None):
    train_conf = f'--layers {l} --neurons {n} --batch {b} --epochs {e} --lr {lr}'
    if save:
        train_conf = f"{train_conf} --save"

    if pretrain_conf:
        train_conf = f"{train_conf} --pretrain {pretrain_conf}"

    data_step = {
        'preprocess': [['labels', 'binarize']],
        'resample': r,
        'resample_amount': ra
    }

    basic_strategy[0][1] = train_conf
    data_steps[0] = data_step
    shared["csv"] = csv
    return main(), f'{train_conf}_{r}_{ra}'


def do_train(csv, l, n, b, e, lr, r, ra, save=False, pretrain_conf=None):
    train_f1, train_conf = only_train(csv, l, n, b, e, lr, r, ra, save, pretrain_conf)
    with open("results_train.csv", 'a') as f:
        f.write(f"{train_f1},{train_conf} {csv}\n")

    return train_f1


def only_pretrain(csv="dataset_warn_full.csv", layers=5, neurons=1024, batch=512, epochs=10, lr=0.1, resample='none',
                  r_amount=0):
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


def full_train(csv, l, n, b, e, lr, r, ra):
    train_conf = f'--layers {l} --neurons {n} --batch {b} --epochs {e} --lr {lr}'
    pre_train_id = f'layers-{l}_neurons-{n}_batch-{b}_epochs-{e}_lr-{lr}_beta-0.0_save-True_{r}_{ra}_1337_{csv[:-4]}'
    train_conf_with_pretrain = f'{train_conf} --pretrain {pre_train_id}'

    data_step = {
        'preprocess': [],
        'resample': r,
        'resample_amount': ra
    }

    basic_strategy[0][1] = f'{train_conf} --save'
    data_steps[0] = data_step
    shared["csv"] = csv
    pretrain_f_measure = main()
    basic_strategy[0][1] = train_conf_with_pretrain
    data_steps[0] = data_step
    shared["csv"] = "graphcodebert_vuln_dataset.csv"
    pretrained_f_measure = main()

    basic_strategy[0][1] = train_conf
    data_steps[0] = data_step
    shared["csv"] = "graphcodebert_vuln_dataset.csv"
    not_pretrained_f_measure = main()

    return pretrain_f_measure, pretrained_f_measure, not_pretrained_f_measure, train_conf_with_pretrain


def do_full_train(csv, l, n, b, e, lr, r, ra):
    pretrain_f1, pretrained_f1, not_pretrained_f1, train_conf = full_train(csv, l, n, b, e, lr, r, ra)
    with open("results2.csv", 'a') as f:
        f.write(f"{pretrain_f1},{pretrained_f1},{not_pretrained_f1},{train_conf}\n")

    return pretrained_f1 - not_pretrained_f1


def touch_path(path):
    path.parent.mkdir(exist_ok=True)
    path.touch(exist_ok=True)
    return path


def full_train_test():
    layers = [7, 5, 3]
    neurons = [2048, 1024, 512]
    batches = [512, 256, 128]
    epochs = [5, 10, 15, 20]
    lrs = [.0001]
    resample = ['down']
    r_amount = [100]
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

                                    pickle_file = Path("pickles") / Path(
                                        "_".join(map(lambda x: str(x), [csv, l, n, b, e, lr, r, ra])))
                                    if pickle_file.exists():
                                        continue

                                    do_full_train(csv, l, n, b, e, lr, r, ra)
                                    touch_path(pickle_file)


def set_fix_values(d, p, names):
    for name in names:
        p[name] = d[f"{name}_list"][int(round(p[name]))]


def set_round_values(p, names):
    for name in names:
        p[name] = int(round(p[name]))


def full_train_bayesian():
    d = {
        "csv_list": ["graphcodebert_warn_dataset_without_minor_and_major.csv",
                     "graphcodebert_warn_dataset_without_minor.csv", "graphcodebert_warn_dataset_full.csv"],
        "n_list": [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "b_list": [128, 256, 512, 1024, 2048, 4096],
        "r_list": ["none", "up", "down"]
    }

    # Create the optimizer. The black box function to optimize is not
    # specified here, as we will call that function directly later on.
    optimizer = BayesianOptimization(f=None,
                                     pbounds={
                                         "csv": [0, 2],
                                         "l": [1, 20],
                                         "n": [0, 7],
                                         "b": [0, 5],
                                         "e": [1, 100],
                                         "lr": [.000001, .1],
                                         "r": [0, 2],
                                         "ra": [1, 100]
                                     },
                                     verbose=2, random_state=1337)
    # Specify the acquisition function (bayes_opt uses the term
    # utility function) to be the upper confidence bounds "ucb".
    # We set kappa = 1.96 to balance exploration vs exploitation.
    # xi = 0.01 is another hyper parameter which is required in the
    # arguments, but is not used by "ucb". Other acquisition functions
    # such as the expected improvement "ei" will be affected by xi.
    utility = UtilityFunction(kind="ucb", kappa=1.96, xi=0.01)

    # Optimization for loop.
    for i in range(25):
        # Get optimizer to suggest new parameter values to try using the
        # specified acquisition function.
        next_point = optimizer.suggest(utility)
        # Force degree from float to int.
        set_fix_values(d, next_point, ["csv", "n", "b", "r"])
        set_round_values(next_point, ["l", "e"])

        if (next_point['r'] == 'none' and next_point['ra'] != 0) or (
                next_point['r'] != 'none' and next_point['ra'] == 0):
            continue

        next_point_values = [next_point[k] for k in ['csv', 'l', 'n', 'b', 'e', 'lr', 'r', 'ra']]
        pickle_file = Path("pickles") / "bayes" / Path("_".join(map(lambda x: str(x), next_point_values)))
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                target = pickle.load(f)

            try:
                optimizer.register(params=next_point, target=target)
            except:
                pass

            print(f"Best result so far:")
            print(optimizer.max)
            continue

        # Evaluate the output of the black_box_function using
        # the new parameter values.
        target = do_full_train(**next_point)
        with open(pickle_file, 'wb') as f:
            pickle.dump(target, f)

        try:
            # Update the optimizer with the evaluation results.
            # This should be in try-except to catch any errors!
            optimizer.register(params=next_point, target=target)
        except:
            pass

        print(f"Best result so far:")
        print(optimizer.max)


# def run_train_bayesian(train_type="train", dataset="dataset_vuln_full.csv", n=100):
#     d = {
#         "n_list": [128, 256, 512, 1024, 2048, 4096],
#         "b_list": [128, 256, 512, 1024, 2048],
#         "r_list": ["none", "up", "down"]
#     }
#
#     # Create the optimizer. The black box function to optimize is not
#     # specified here, as we will call that function directly later on.
#     optimizer = BayesianOptimization(f=None,
#                                      pbounds={
#                                          "l": [3, 7],
#                                          "n": [0, 5],
#                                          "b": [0, 4],
#                                          "e": [1, 20],
#                                          "lr": [.000001, .1],
#                                          "r": [0, 2],
#                                          "ra": [0, 100]
#                                      },
#                                      verbose=2, random_state=1337)
#
#     # Specify the acquisition function (bayes_opt uses the term
#     # utility function) to be the upper confidence bounds "ucb".
#     # We set kappa = 1.96 to balance exploration vs exploitation.
#     # xi = 0.01 is another hyper parameter which is required in the
#     # arguments, but is not used by "ucb". Other acquisition functions
#     # such as the expected improvement "ei" will be affected by xi.
#     utility = UtilityFunction(kind="ucb", kappa=1.96, xi=0.01)
#
#     best_f1_score = -1
#
#     # Optimization for loop.
#     for i in range(n):
#         print(f'{i + 1}. iter')
#         # Get optimizer to suggest new parameter values to try using the
#         # specified acquisition function.
#         next_point = optimizer.suggest(utility)
#         # Force degree from float to int.
#         set_fix_values(d, next_point, ["n", "b", "r"])
#         set_round_values(next_point, ["l", "e"])
#
#         if next_point['r'] == 'none' or next_point['ra'] == 0:
#             next_point['r'] = 'none'
#             next_point['ra'] = 0
#
#         next_point_values = [next_point[k] for k in ['l', 'n', 'b', 'e', 'lr', 'r', 'ra']]
#         pickle_file = Path("pickles") / f"bayes_{train_type}" / Path(
#             f"{dataset}_" + "_".join(map(lambda x: str(x), next_point_values)))
#         if pickle_file.exists():
#             with open(pickle_file, 'rb') as f:
#                 target = pickle.load(f)
#
#         else:
#             # Evaluate the output of the black_box_function using
#             # the new parameter values.
#             target = do_train(csv=dataset, **next_point)
#             with open(pickle_file, 'wb') as f:
#                 pickle.dump(target, f)
#
#         if target > best_f1_score:
#             best_f1_score = target
#
#         try:
#             # Update the optimizer with the evaluation results.
#             # This should be in try-except to catch any errors!
#             optimizer.register(params=next_point, target=target)
#
#         except:
#             pass
#
#         print(f"Best result so far:")
#         print(optimizer.max)
#
#     return best_f1_score


def run_fine_tune_bayesian(train_type="fine_tune", pretrain_conf=None, dataset="dataset_vuln_full.csv", n=100):
    d = {
        "b_list": [128, 256, 512, 1024, 2048],
        "r_list": ["none", "up", "down"]
    }

    # Create the optimizer. The black box function to optimize is not
    # specified here, as we will call that function directly later on.
    optimizer = BayesianOptimization(f=None,
                                     pbounds={
                                         "b": [0, 4],
                                         "e": [1, 20],
                                         "lr": [.000001, .1],
                                         "r": [0, 2],
                                         "ra": [0, 100]
                                     },
                                     verbose=2, random_state=1337)

    # Specify the acquisition function (bayes_opt uses the term
    # utility function) to be the upper confidence bounds "ucb".
    # We set kappa = 1.96 to balance exploration vs exploitation.
    # xi = 0.01 is another hyper parameter which is required in the
    # arguments, but is not used by "ucb". Other acquisition functions
    # such as the expected improvement "ei" will be affected by xi.
    utility = UtilityFunction(kind="ucb", kappa=1.96, xi=0.01)

    best_f1_score = -1

    # Optimization for loop.
    for i in range(n):
        print(f'{i + 1}. iter')
        # Get optimizer to suggest new parameter values to try using the
        # specified acquisition function.
        next_point = optimizer.suggest(utility)
        # Force degree from float to int.
        set_fix_values(d, next_point, ["b", "r"])
        set_round_values(next_point, ["e"])

        if next_point['r'] == 'none' or next_point['ra'] == 0:
            next_point['r'] = 'none'
            next_point['ra'] = 0

        next_point['l'] = 0
        next_point['n'] = 0
        next_point_values = [next_point[k] for k in ['b', 'e', 'lr', 'r', 'ra']]
        pre_train_conf_values = re.split(r'[-_]', pretrain_conf)
        values = [v for i, v in enumerate(pre_train_conf_values) if
                  i in ([1, 3, 5, 7, 9, 11] + list(range(14, len(pre_train_conf_values))))]
        pickle_file_name = Path(f"{dataset}_" + "_".join(map(lambda x: str(x), next_point_values)) + "_".join(values))
        pickle_file = Path("pickles") / f"bayes_{train_type}" / pickle_file_name
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                target = pickle.load(f)

        else:
            # Evaluate the output of the black_box_function using
            # the new parameter values.
            target = do_train(csv=dataset, **next_point, pretrain_conf=pretrain_conf)
            with open(pickle_file, 'wb') as f:
                pickle.dump(target, f)

        if target > best_f1_score:
            best_f1_score = target

        try:
            # Update the optimizer with the evaluation results.
            # This should be in try-except to catch any errors!
            optimizer.register(params=next_point, target=target)

        except:
            pass

        print(f"Best result so far:")
        print(optimizer.max)

    return best_f1_score


def get_pre_train_conf_id(l, n, b, e, lr, r, ra, csv):
    return f'layers-{l}_neurons-{n}_batch-{b}_epochs-{e}_lr-{lr}_beta-0.0_save-True_{r}_{ra}_1337_{csv[:-4]}'


def run_transfer_learning_bayesian(train_type="transfer_learning", dataset="graphcodebert_warn_dataset_full.csv",
                                   fine_tune_dataset="", n=100):
    d = {
        "n_list": [128, 256, 512, 1024, 2048, 4096],
        "b_list": [128, 256, 512, 1024, 2048],
        "r_list": ["none", "up", "down"]
    }

    # Create the optimizer. The black box function to optimize is not
    # specified here, as we will call that function directly later on.
    optimizer = BayesianOptimization(f=None,
                                     pbounds={
                                         "l": [3, 7],
                                         "n": [0, 5],
                                         "b": [0, 4],
                                         "e": [1, 20],
                                         "lr": [.000001, .1],
                                         "r": [0, 2],
                                         "ra": [0, 100]
                                     },
                                     verbose=2, random_state=1337)

    # Specify the acquisition function (bayes_opt uses the term
    # utility function) to be the upper confidence bounds "ucb".
    # We set kappa = 1.96 to balance exploration vs exploitation.
    # xi = 0.01 is another hyper parameter which is required in the
    # arguments, but is not used by "ucb". Other acquisition functions
    # such as the expected improvement "ei" will be affected by xi.
    utility = UtilityFunction(kind="ucb", kappa=1.96, xi=0.01)

    best_f1 = -1
    # Optimization for loop.
    for i in range(n):
        print(f'{i + 1}. iter')
        # Get optimizer to suggest new parameter values to try using the
        # specified acquisition function.
        next_point = optimizer.suggest(utility)
        # Force degree from float to int.
        set_fix_values(d, next_point, ["n", "b", "r"])
        set_round_values(next_point, ["l", "e"])

        if next_point['r'] == 'none' or next_point['ra'] == 0:
            next_point['r'] = 'none'
            next_point['ra'] = 0

        next_point_values = [next_point[k] for k in ['l', 'n', 'b', 'e', 'lr', 'r', 'ra']]
        pickle_file = Path("pickles") / f"bayes_{train_type}" / Path(
            f"{dataset}_" + "_".join(map(lambda x: str(x), next_point_values)))
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                target = pickle.load(f)

        else:
            # Evaluate the output of the black_box_function using
            # the new parameter values.
            pre_train_f1 = do_train(csv=dataset, **next_point, save=True)
            pre_train_id = get_pre_train_conf_id(csv=dataset, **next_point)
            best_fine_tune_f1 = run_fine_tune_bayesian(pretrain_conf=pre_train_id, dataset=fine_tune_dataset)
            target = best_fine_tune_f1
            with open(pickle_file, 'wb') as f:
                pickle.dump(target, f)

        if target > best_f1:
            best_f1 = target

        try:
            # Update the optimizer with the evaluation results.
            # This should be in try-except to catch any errors!
            optimizer.register(params=next_point, target=target)

        except:
            pass

        print(f"Best result so far:")
        print(optimizer.max)


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
                                    pickle_file = Path("pickles") / Path(
                                        "_".join(map(lambda x: str(x), [csv, l, n, b, e, lr, r, ra])))
                                    if pickle_file.exists():
                                        continue

                                    res, conf = only_pretrain(csv, l, n, b, e, lr, r, ra)
                                    with open("results_pretrain.csv", 'a') as f:
                                        f.write(f"{res},{conf} {csv}\n")

                                    with open(touch_path(pickle_file), "wb") as f:
                                        pickle.dump(pickle_file, f)


def run_train_bayesian(train_type="ossf_test", dataset="graphcodebert_warn_dataset_full.csv", n=100):
    d = {
        "n_list": [128, 256, 512, 1024, 2048, 4096],
        "b_list": [128, 256, 512, 1024, 2048],
        "r_list": ["none", "up", "down"]
    }

    # Create the optimizer. The black box function to optimize is not
    # specified here, as we will call that function directly later on.
    optimizer = BayesianOptimization(f=None,
                                     pbounds={
                                         "l": [3, 7],
                                         "n": [0, 5],
                                         "b": [0, 4],
                                         "e": [1, 20],
                                         "lr": [.000001, .1],
                                         "r": [0, 2],
                                         "ra": [0, 100]
                                     },
                                     verbose=2, random_state=1337)

    # Specify the acquisition function (bayes_opt uses the term
    # utility function) to be the upper confidence bounds "ucb".
    # We set kappa = 1.96 to balance exploration vs exploitation.
    # xi = 0.01 is another hyper parameter which is required in the
    # arguments, but is not used by "ucb". Other acquisition functions
    # such as the expected improvement "ei" will be affected by xi.
    utility = UtilityFunction(kind="ucb", kappa=1.96, xi=0.01)

    best_f1 = -1
    # Optimization for loop.
    for i in range(n):
        print(f'{i + 1}. iter')
        # Get optimizer to suggest new parameter values to try using the
        # specified acquisition function.
        next_point = optimizer.suggest(utility)
        # Force degree from float to int.
        set_fix_values(d, next_point, ["n", "b", "r"])
        set_round_values(next_point, ["l", "e"])

        if next_point['r'] == 'none' or next_point['ra'] == 0:
            next_point['r'] = 'none'
            next_point['ra'] = 0

        next_point_values = [next_point[k] for k in ['l', 'n', 'b', 'e', 'lr', 'r', 'ra']]
        pickle_file = Path("pickles") / f"bayes_{train_type}" / f'{dataset}_{"_".join(map(lambda x: str(x), next_point_values))}'
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                target = pickle.load(f)

        else:
            # Evaluate the output of the black_box_function using
            # the new parameter values.
            target = do_train(csv=dataset, **next_point, save=True)
            with open(pickle_file, 'wb') as f:
                pickle.dump(target, f)

        if target > best_f1:
            best_f1 = target

        try:
            # Update the optimizer with the evaluation results.
            # This should be in try-except to catch any errors!
            optimizer.register(params=next_point, target=target)

        except:
            pass

        print(f"Best result so far:")
        print(optimizer.max)


if __name__ == '__main__':
    # main()
    # full_train_test()
    # run_train_bayesian(train_type="pretrain", dataset="dataset_warn_full.csv", n=200)
    run_train_bayesian(dataset="train_balanced_features_data.csv", n=100)
    # pre_train_test()
