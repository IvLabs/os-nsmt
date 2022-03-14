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
from common.utils.data import ForeverDataIterator
from data_helper import setup_datasets

import wandb
os.environ['WANDB_API_KEY'] = '93b09c048a71a2becc88791b28475f90622b0f63'
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Total GPUs Used:", torch.cuda.device_count())
i = 0
print("Hardwares Used: ")
while(i < torch.cuda.device_count()):
  print(torch.cuda.get_device_name(i))
  i = i + 1


print(up1)
DATA_DIR = 'data/office-home'

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

    hparams["model"] = args.model
    if args.batch_size > 0:
        hparams["batch_size"] = args.batch_size

    print("HParams:")
    for k, v in sorted(hparams.items()):
        print("\t{}: {}".format(k, v))

    if args.wandb:
        wandb.config.update(args)
        wandb.config.update(hparams)
    if args.tensorboard:
        writer.add_hparams({'seed': args.seed, 'steps': args.steps, 'batch_size': hparams['batch_size'], 'mixup': hparams['mixup']}, {'proto_loss': 0.0, 'proto_accuracy': 0.0})
    return hparams


def _setup_datasets(args, hparams):
    args.root = "./data"
    num_classes, train_source_loader, train_target_loader, _, _  = setup_datasets(args, False)
    splits = train_source_loader + train_target_loader
    trn_d = sorted([loader.dataset.domain_index for loader in splits])
    # train_loaders = [splits[i] for i in trn_d]
    train_loaders = sorted(splits, key=lambda x: x.dataset.domain_index)
    steps_per_epoch = min(len(env.dataset) / hparams["batch_size"] for env in splits)

    return train_loaders, steps_per_epoch, trn_d


def _setup_algorithm(
    args, device, hparams, trn_d
):
    """ Wrapper function to set up algorithm from args and hyperparams. """
    alg_args = args.algorithm.split(":")

    algorithm_name = alg_args[0]

    if "mixup" in alg_args:
        hparams["mixup"] = 1.0

    algorithm_class = algorithms_proto.get_algorithm_class(algorithm_name)

    # setting prototype num_step hyperparameter
    if algorithm_name.startswith('Proto'):
        rounds_proto = math.ceil(len(trn_d) / hparams["proto_domains_per_iter"])
        num_proto_steps = int(
            math.ceil(hparams["num_proto_extraction_points"]  * 1.0 / hparams["batch_size"])
        )
        hparams["n_steps_proto"] = num_proto_steps * rounds_proto

    # calling appropriate algorithm
    algorithm = algorithm_class(hparams)

    # moving to appropriate device
    algorithm.to(device)

    return algorithm


def train_prototype(
    args,
    hparams,
    algorithm,
    train_loaders,
    steps_per_epoch,
    device,
):
    """ Wrapper over prototypical training procedure. """

    # train_loaders = [iter(x) for x in train_loaders]
    train_loaders = [ForeverDataIterator(x) for x in train_loaders]
    num_train_domains = len(train_loaders)

    proto_checkpoint_vals = collections.defaultdict(lambda: [])

    n_prototype_steps = 8000
    if args.num_proto_steps > 0:
        n_prototype_steps = args.num_proto_steps
    print("Training prototype for %d steps..." % (n_prototype_steps))
    rounds = math.ceil(num_train_domains / hparams["proto_domains_per_iter"]) # rounds = 1
    proto_checkpoint_freq = args.checkpoint_freq
    dpi = hparams["proto_domains_per_iter"] # dpi=4

    algorithm.prototyper.train()

    start_step = hparams['start_step']
    for p_step in range(start_step, n_prototype_steps):
        step_start_time = time.time()

        step_vals = collections.defaultdict(lambda: [])
        minibatches = _get_minibatches(train_loaders)

        random.shuffle(minibatches)

        for i in range(rounds): # round = 1
            end_idx = min(len(minibatches), dpi * (i + 1)) # dpi: 4, len(minibatches): 4, end_idx = 4
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
        if args.wandb:
            wandb.log({'proto step': p_step, 'proto_loss': proto_checkpoint_vals["proto_loss"][-1], 'proto_accuracy': proto_checkpoint_vals["proto_acc"][-1]})
        if args.tensorboard:
            writer.add_scalar('proto_loss', proto_checkpoint_vals["proto_loss"][-1], p_step)
            writer.add_scalar('proto_accuracy', proto_checkpoint_vals["proto_acc"][-1], p_step)

        proto_checkpoint_vals["step_time"].append(step_end_time)

        if p_step % proto_checkpoint_freq == 0 and p_step > 0:
            results = {"proto_step": p_step, "epoch": p_step / steps_per_epoch}

            for key, val in proto_checkpoint_vals.items():
                results[key] = np.mean(val)

            epochs_path = os.path.join(args.output_dir, "proto_results.jsonl")
            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            checkpoint_file = os.path.join(args.output_dir, "prototype_%d.pth" % (p_step))
            algorithm.save_state(checkpoint_file, p_step)

            proto_checkpoint_vals = collections.defaultdict(lambda: [])

    final_proto_chkpt = os.path.join(args.output_dir, "prototype_final.pth")
    algorithm.save_state(final_proto_chkpt, n_prototype_steps-1)


def compute_prototype(
    algorithm, data_loader, device
):
    """ This function takes in a data loader and averages the prototype
        feature over all domains in the data loader """

    data_loader = [iter(x) for x in data_loader]
    # data_loader = [ForeverDataIterator(x) for x in data_loader]

    prototypes = {}

    algorithm.prototyper.eval()
    with torch.no_grad():

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
                "Prototype extraction, domain %d of %d done..."
                % (idx + 1, len(data_loader))
            )

    return prototypes


def main(args):
    """ Main function calling prototype training functions as subroutine."""
    if args.wandb:
        wandb.init(project = 'degaa', entity = 'abd1')
        wandb.run.name = 'ProtoType'
    if args.tensorboard:
        global writer
        writer = SummaryWriter(os.path.join(args.output_dir,"tensorboard"))

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, "out.txt"))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, "err.txt"))

    hparams = _setup_hparams(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # device = "cpu"

    (
        train_loaders, # list of 4 Dataloader: OpenSetDataset with 26 classes of Art, Product, Clipart, Real_World
        steps_per_epoch, 
        trn_d,
    ) = _setup_datasets(args, hparams)

    #dataset = dalib.domainbed.datasets.OfficeHome
    #train_loaders = len(): 4
    # steps_per_epoch: 161.83333333333334
    # trn_d = [0, 1, 2, 3]

    algorithm = _setup_algorithm(
        args, device, hparams, trn_d
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
                if args.resume is not None:
                    args.resume = False
            if not os.path.exists(prototypes_file):
                do_prototype_extraction = True

        if do_prototype_training:
            print("::: PROTOTYPE TRAINING :::")
            print("==========================")
            algorithm.init_prototype_training()
            hparams["start_step"] = 0
            if args.resume is not None:
                proto_model = os.path.join(args.output_dir, "prototype_%d.pth" % (args.resume))
                hparams['start_step'] = algorithm.load_state(proto_model)
                print("::: RESUMING TRAINING FROM STEP %d :::" % (hparams["start_step"]))
                print("======================================")
            train_prototype(
                args,
                hparams,
                algorithm,
                train_loaders,
                steps_per_epoch,
                device,
            )
        else:
            print("::: LOADING PROTOTYPE MODEL :::")
            print("===============================")

            proto_model = os.path.join(hparams["proto_model"], "prototype_final.pth")
            algorithm.load_state(proto_model)

        # extract prototypes
        if do_prototype_extraction:
            print("::: PROTOTYPE EXTRACTION :::")
            print("============================")

            prototypes = compute_prototype(algorithm, train_loaders, device)

            # now save prototypes to file
            prototype_file = os.path.join(args.output_dir, "prototypes.pth")
            torch.save(prototypes, prototype_file)
        else:
            print("::: LOADING PROTOTYPE FROM FILE :::")
            print("===================================")

            prototypes_file = os.path.join(hparams["proto_model"], "prototypes.pth")
            prototypes = torch.load(prototypes_file)


    ### COMPLETED ###
    if args.tensorboard:
        writer.close()

    with open(os.path.join(args.output_dir, "done"), "w") as f:
        f.write("done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Domain Embeddings")
    parser.add_argument("--dataset", type=str, default="OfficeHome")
    parser.add_argument("-s", "--source", help="source domain(s)", default="Ar,Pr")
    parser.add_argument("-t", "--target", help="target domain(s)", default="Cl,Rw")
    parser.add_argument("--algorithm", type=str, default="Proto")
    parser.add_argument("--hparams", type=str, help="JSON-serialized hparams dict")
    parser.add_argument("--hparams_seed", type=int, default=0, help='Seed for random hparams (0 means "default hparams")',)
    parser.add_argument("--trial_seed", type=int, default=0, help="Trial number (used for seeding split_dataset and " "random_hparams).",)
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument("--steps", type=int, default=8000, help="Number of steps. Default is dataset-dependent.")
    parser.add_argument("--checkpoint_freq", type=int, default=100, help="Checkpoint every N steps. Default is dataset-dependent.")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_proto_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./protoruns/temp", required=False)
    parser.add_argument("--proto_dir", type=str, required=False)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--resume", type=int, required=False, help='enables resuming')
    parser.add_argument("--wandb", action='store_true', help='enables wandb logging')
    parser.add_argument("--tensorboard", action='store_true', help='enables tensorboard logging')
    args = parser.parse_args()

    # args.data_dir, args.model_dir = get_data_model_dir()
    args.data_dir = DATA_DIR
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    main(args)

