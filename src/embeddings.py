#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import math
import os
import sys
from os.path import dirname

sys.path.append(os.getcwd())
up1 = dirname(dirname(__file__))
sys.path.insert(0, up1) 

import random
import time

import numpy as np
import torch
import torch.utils.data
from dalib.domainbed import algorithms_proto, datasets, hparams_registry
from dalib.domainbed.lib import misc
from dalib.domainbed.lib.fast_data_loader import FastDataLoader
import wandb
wandb.init(project = 'degaa')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Total GPUs Used:", torch.cuda.device_count())
i = 0
print("Hardwares Used: ")
while(i < torch.cuda.device_count()):
  print(torch.cuda.get_device_name(i))
  i = i + 1

wandb.run.name = 'DE_OfficeHome'

print(up1)
DATA_DIR = 'data/office-home'
MODEL_DIR = up1 + '/models'
OUTPUT_DIR = up1 + '/outputs'

def _get_minibatches(data_loader):
    """ Wrapper around minibatch loader"""

    minibatches = [next(x) for x in data_loader]
    return [(x, y, i) for i, (x, y) in enumerate(minibatches)]


def _setup_hparams(args):
    """ Wrapper function to get hyperparameters from input args. """
    print("Args:")
    for k, v in sorted(vars(args).items()):
        print("\t{}: {}".format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(
            args.algorithm,
            args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed),
        )
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    
    if args.proto_dir:
        
        if os.path.exists(args.proto_dir)\
                and os.path.exists(args.proto_dir+'/prototype_final.pth'):
            fp = args.proto_dir
            hparams["proto_model"] = fp
            hparams["train_prototype"] = False

    hparams["model_dir"] = args.model_dir
    hparams["model"] = args.model
    if args.batch_size > 0:
        hparams["batch_size"] = args.batch_size

    print("HParams:")
    for k, v in sorted(hparams.items()):
        print("\t{}: {}".format(k, v))

    # wandb.config.update(args)
    # wandb.config.update(hparams)
    return hparams


def _setup_datasets(args, hparams):
    """ Wrapper function to set up datasets from args and hyperparams. """

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # storing training and test environments

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    trn_d, tst_d = [], []
    for i in range(len(in_splits)):
        if i not in args.test_envs:
            trn_d.append(i)
        else:
            tst_d.append(i)

    train_loaders = [
        FastDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=hparams["batch_size"],
            num_workers=dataset.N_WORKERS,
            length=FastDataLoader.INFINITE,
        )
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs
    ]

    eval_loaders = [
        FastDataLoader(
            dataset=env,
            weights=None,
            batch_size=64,
            num_workers=dataset.N_WORKERS,
            length=FastDataLoader.EPOCH,
        )
        for env, _ in (in_splits + out_splits)
    ]
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]

    steps_per_epoch = min(len(env) / hparams["batch_size"] for env, _ in in_splits)

    return (
        dataset,
        train_loaders,
        eval_loaders,
        eval_weights,
        eval_loader_names,
        steps_per_epoch,
        trn_d,
        tst_d,
    )

def _setup_algorithm(
    args, algorithm_dict, device, dataset, hparams, trn_d, tst_d
):
    """ Wrapper function to set up algorithm from args and hyperparams. """
    alg_args = args.algorithm.split(":")

    algorithm_name = alg_args[0]

    if "mixup" in alg_args:
        hparams["mixup"] = 1.0

    # try:
    #     algorithm_class = algorithms.get_algorithm_class(algorithm_name)
    # except NotImplementedError:
    algorithm_class = algorithms_proto.get_algorithm_class(algorithm_name)

    # setting prototype num_step hyperparameter
    if algorithm_name.startswith('Proto'):
        rounds_proto = math.ceil(len(trn_d) / hparams["proto_domains_per_iter"])
        num_proto_steps = int(
            math.ceil(dataset.NUM_PROTO_EXTRACTION_POINTS * 1.0 / hparams["batch_size"])
        )
        hparams["n_steps_proto"] = num_proto_steps * rounds_proto

    # setting regular training num_step hyperparameter
    rounds = math.ceil(len(trn_d) / hparams["domains_per_iter"])
    num_steps = int(math.ceil(dataset.N_STEPS))
    hparams["n_steps"] = num_steps * rounds

    # calling appropriate algorithm
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(args.test_envs),
        hparams,
    )

    # TODO: resume training from previous state_dict (DomainBed)
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    # moving to appropriate device
    algorithm.to(device)

    return algorithm


def train_prototype(
    args,
    hparams,
    dataset,
    algorithm,
    train_loaders,
    steps_per_epoch,
    device,
):
    """ Wrapper over prototypical training procedure. """

    train_loaders = [iter(x) for x in train_loaders]
    num_train_domains = len(train_loaders)

    proto_checkpoint_vals = collections.defaultdict(lambda: [])

    n_prototype_steps = dataset.NUM_PROTO_STEPS
    print("Training prototype for %d steps..." % (n_prototype_steps))
    rounds = math.ceil(num_train_domains / hparams["proto_domains_per_iter"])
    proto_checkpoint_freq = args.checkpoint_freq or dataset.PROTO_CHECKPOINT_FREQ
    dpi = hparams["proto_domains_per_iter"]

    for p_step in range(n_prototype_steps):
        step_start_time = time.time()

        step_vals = collections.defaultdict(lambda: [])
        minibatches = _get_minibatches(train_loaders)

        random.shuffle(minibatches)

        for i in range(rounds):
            end_idx = min(len(minibatches), dpi * (i + 1))
            minibatches_device = [
                (x.to(device), y.to(device))
                for x, y, _ in minibatches[dpi * i : end_idx]
            ]
            stv_x = algorithm.prototype_update(minibatches_device)
            for key, val in stv_x.items():
                step_vals[key].append(val)

        for k, v in step_vals.items():
            proto_checkpoint_vals[k].append(sum(v) * 1.0 / len(v))

        step_end_time = time.time() - step_start_time

        if p_step % hparams["proto_log_train_step"] == 0:
            print(
                "prototype step %d, loss %.2f, accuracy %.2f, time %.2f"
                % (
                    p_step,
                    proto_checkpoint_vals["proto_loss"][-1],
                    proto_checkpoint_vals["proto_acc"][-1],
                    step_end_time,
                )
            )
            wandb.log({'proto step': p_step, 'proto_loss': proto_checkpoint_vals["proto_loss"][-1], 'proto_accuracy': proto_checkpoint_vals["proto_acc"][-1]})

        proto_checkpoint_vals["step_time"].append(step_end_time)

        if p_step % proto_checkpoint_freq == 0 and p_step > 0:
            results = {"proto_step": p_step, "epoch": p_step / steps_per_epoch}

            for key, val in proto_checkpoint_vals.items():
                results[key] = np.mean(val)

            epochs_path = os.path.join(args.output_dir, "proto_results.jsonl")
            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            checkpoint_file = os.path.join(
                args.output_dir, "prototype_%d.pth" % (p_step)
            )
            #algorithm.save_prototype(checkpoint_file)

            proto_checkpoint_vals = collections.defaultdict(lambda: [])

    final_proto_chkpt = os.path.join(args.output_dir, "prototype_final.pth")
    algorithm.save_prototype(final_proto_chkpt)


def compute_prototype(
    algorithm, data_loader, num_steps, device, hparams, phase="train"
):
    """ This function takes in a data loader and averages the prototype
        feature over all domains in the data loader """

    data_loader = [iter(x) for x in data_loader]

    prototypes = {}

    algorithm.featurizer.eval()
    with torch.no_grad():

        if phase == "train":
            for s in range(1, num_steps + 1):

                minibatches = _get_minibatches(data_loader)

                for idx, (x, _, _) in enumerate(minibatches):

                    x = x.to(device)
                    x_avg = algorithm.compute_average_prototype(x)

                    if idx not in prototypes:
                        prototypes[idx] = x_avg
                    else:
                        prototypes[idx] = (prototypes[idx] * s + x_avg) / (s + 1)

                if s % hparams["proto_log_avg_step"] == 0:
                    print(
                        "Prototype (%s) extraction, step %d of %d done..."
                        % (phase, s, num_steps)
                    )
        else:
            # proceed one by one
            for idx, loader in enumerate(data_loader):
                np = 0
                for (x, _) in loader:

                    bs = x.shape[0]
                    np += bs
                    x = x.to(device)
                    x_avg = algorithm.compute_average_prototype(x)

                    if idx not in prototypes:
                        prototypes[idx] = x_avg
                    else:
                        prototypes[idx] = (prototypes[idx] * np + x_avg * bs) / (
                            np + bs
                        )

                print(
                    "Prototype (%s) extraction, domain %d of %d done..."
                    % (phase, idx + 1, len(data_loader))
                )

    return prototypes


def train_main(
    args,
    hparams,
    dataset,
    algorithm,
    train_loaders,
    device,
    steps_per_epoch,
    eval_loader_names,
    eval_loaders,
):
    """ Main training function. """

    # initializing details
    is_proto = "Proto" in args.algorithm

    start_step = 0
    checkpoint_vals = collections.defaultdict(lambda: [])
    train_loaders = [iter(x) for x in train_loaders]
    num_train_domains = len(train_loaders)

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    last_results_keys = None
    rounds = math.ceil(num_train_domains / hparams["domains_per_iter"])
    dpi = hparams["domains_per_iter"]

    num_val_steps = int(
        math.ceil(dataset.NUM_PROTO_EXTRACTION_POINTS * 1.0 / hparams["batch_size"])
    )

    for step in range(start_step, n_steps):
        step_start_time = time.time()

        step_vals = collections.defaultdict(lambda: [])
        minibatches = _get_minibatches(train_loaders)

        for i in range(rounds):
            end_idx = min(len(minibatches), dpi * (i + 1))
            minibatches_device = [
                (x.to(device), y.to(device))
                for x, y, _ in minibatches[dpi * i : end_idx]
            ]
            if is_proto:
                stv_x = algorithm.update(minibatches_device, device=device)
            else:
                stv_x = algorithm.update(minibatches_device)

            for key, val in stv_x.items():
                step_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if step % checkpoint_freq == 0:
            results = {"step": step, "epoch": step / steps_per_epoch}

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            accs = []
            evals = zip(eval_loader_names, eval_loaders)
            ne = int(len(eval_loaders) / 2)
            for idx, (name, loader) in enumerate(evals):
                if is_proto:
                    acc = misc.accuracy(algorithm, loader, device, idx % ne)
                else:
                    acc = misc.accuracy(algorithm, loader, device, -1)
                results[name + "_acc"] = acc
                wandb.log({name + "_acc": acc})

                accs.append(acc)
            wandb.log({'main_loss': results['loss']})
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys and step % (checkpoint_freq * 100) == 0:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            if step % (checkpoint_freq * 100) == 0:
                misc.print_row([results[key] for key in results_keys], colwidth=12)

            results.update({"hparams": hparams, "args": vars(args)})

            epochs_path = os.path.join(args.output_dir, "results.jsonl")
            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")
            model_path = os.path.join(args.output_dir, "model_%d.pth" % (step))
            #algorithm.save_model(model_path)

            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

    model_path = os.path.join(args.output_dir, "model_final.pth")
    algorithm.save_model(model_path)


def main(args):
    """ Main function calling prototype training functions as subroutine."""
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, "out.txt"))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, "err.txt"))

    hparams = _setup_hparams(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    (
        dataset,
        train_loaders,
        eval_loaders,
        eval_weights,
        eval_loader_names,
        steps_per_epoch,
        trn_d,
        tst_d,
    ) = _setup_datasets(args, hparams)

    algorithm = _setup_algorithm(
        args, algorithm_dict, device, dataset, hparams, trn_d, tst_d
    )

    if "Proto" in args.algorithm:
        # method that trains prototypical network
        do_prototype_training =\
            hparams["proto_model"] is None or hparams["train_prototype"]

        do_prototype_extraction = do_prototype_training

        if not do_prototype_training:
            proto_model = os.path.join(hparams["proto_model"], "prototype_final.pth")
            prototypes_file = os.path.join(hparams["proto_model"], "prototypes.pth")

            if not os.path.exists(proto_model):
                do_prototype_training = True
            if not os.path.exists(prototypes_file):
                do_prototype_extraction = True

        if do_prototype_training:
            print("::: PROTOTYPE TRAINING :::")
            print("==========================")
            algorithm.init_prototype_training()
            train_prototype(
                args,
                hparams,
                dataset,
                algorithm,
                train_loaders,
                steps_per_epoch,
                device,
            )
        else:
            print("::: LOADING PROTOTYPE MODEL :::")
            print("===============================")

            proto_model = os.path.join(hparams["proto_model"], "prototype_final.pth")
            algorithm.load_prototype(proto_model)

        # extract prototypes
        if do_prototype_extraction:
            num_proto_steps = int(
                math.ceil(
                    dataset.NUM_PROTO_EXTRACTION_POINTS * 1.0 / hparams["batch_size"]
                )
            )
            print("::: PROTOTYPE EXTRACTION FOR %d STEPS :::" % (num_proto_steps))
            print("=========================================")

            # replace prototypes from training domains with training data

            prototypes = {}

            # domainbed, runs only one loader since it tests on training data
            # as well
            raw_prototypes = compute_prototype(
                algorithm,
                eval_loaders,
                num_proto_steps,
                device,
                hparams,
                phase="test",
            )

            nl = len(eval_loaders)
            for i in range(int(nl / 2)):
                bias = int(nl / 2) if i in tst_d else 0
                prototypes[i] = raw_prototypes[i + bias]

            # now save prototypes to file
            prototype_file = os.path.join(args.output_dir, "prototypes.pth")
            torch.save(prototypes, prototype_file)
        else:
            print("::: LOADING PROTOTYPE FROM FILE :::")
            print("===================================")

            prototypes_file = os.path.join(hparams["proto_model"], "prototypes.pth")
            prototypes = torch.load(prototypes_file)

        algorithm.attach_prototypes(prototypes)
        algorithm.init_main_training(hparams)

    # now doing main training

    print("::: MAIN TRAINING :::")
    print("=====================")
    train_main(
        args,
        hparams,
        dataset,
        algorithm,
        train_loaders,
        device,
        steps_per_epoch,
        eval_loader_names,
        eval_loaders,
    )

    with open(os.path.join(args.output_dir, "done"), "w") as f:
        f.write("done")


if __name__ == "__main__":

    # dataset = datasets.__dict__[args.dataset]
    # source_dataset = open_set(dataset, source = True)
    # target_dataset = open_set(dataset, source = False)

    parser = argparse.ArgumentParser(description="Domain Embeddings")
    parser.add_argument("--dataset", type=str, default="OfficeHome")
    parser.add_argument("--algorithm", type=str, default="Proto")
    parser.add_argument("--hparams", type=str, help="JSON-serialized hparams dict")
    parser.add_argument(
        "--hparams_seed",
        type=int,
        default=0,
        help='Seed for random hparams (0 means "default hparams")',
    )
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and " "random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps",
        type=int,
        default=8000,
        help="Number of steps. Default is dataset-dependent.",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=[-1])
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--proto_dir", type=str, required=False)
    args = parser.parse_args()

    # args.data_dir, args.model_dir = get_data_model_dir()
    args.data_dir = DATA_DIR
    args.model_dir = MODEL_DIR
    # args.output_dir = OUTPUT_DIR
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #wandb.init(project="DomainEmbeddings")

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    main(args)


# Dataset link: https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?usp=sharing&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw
