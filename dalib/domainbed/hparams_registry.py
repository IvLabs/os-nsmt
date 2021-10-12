#!/usr/bin/env python3

import numpy as np


def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    RESNET_DATASETS = [
        "VLCS",
        "PACS",
        "PACS_Debug",
        "PACS_Modified",
        "OfficeHome",
        "TerraIncognita",
        "DomainNet",
        "YFCC_Geo",
        "YFCC_Debug",
    ]

    hparams = {}

    hparams['data_augmentation']= (True, lambda r: True)
    hparams['nonlinear_classifier'] = (False, lambda r: False)
    hparams["dataset"] = (dataset, dataset)
    hparams["domains_per_iter"] = (4, 4)
    hparams["data_parallel"] = (True, True)

    if dataset in RESNET_DATASETS:
        hparams["lr"] = (1e-4, 10 ** random_state.uniform(-5.5, -3.5))
        hparams["batch_size"] = (12, 12)
    else:
        hparams["lr"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
        hparams["batch_size"] = (64, int(2 ** random_state.uniform(3, 9)))

    if dataset in ["YFCC_Geo"]:
        hparams["batch_size"] = (100, 100)

    if dataset in ["ColoredMNIST", "RotatedMNIST"]:
        hparams["weight_decay"] = (0.0, 0.0)
    else:
        hparams["weight_decay"] = (1e-4, 10 ** random_state.uniform(-4, -4))

    hparams["class_balanced"] = (False, False)

    if algorithm in ["DANN", "CDANN"]:

        if dataset in RESNET_DATASETS:
            hparams["lr_g"] = (5e-5, 5e-5)
            hparams["lr_d"] = (5e-5, 5e-5)
        else:
            hparams["lr_g"] = (1e-3, 1e-3)
            hparams["lr_d"] = (1e-3, 1e-3)

        if dataset in ["ColoredMNIST", "RotatedMNIST"]:
            hparams["weight_decay_g"] = (0.0, 0.0)
        else:
            hparams["weight_decay_g"] = (0.0, 0.0)

        hparams["lambda"] = (1.0, 1.0)
        hparams["weight_decay_d"] = (0.0, 0.0)
        hparams["d_steps_per_g_step"] = (1, 1)
        hparams["grad_penalty"] = (0.0, 0.0)
        hparams["beta1"] = (0.5, 0.5)

    if algorithm.startswith("Proto"):
        # optimizer params
        hparams["proto_lr"] = (1e-6, 10 ** random_state.uniform(-7, -5))
        hparams["proto_weight_decay"] = (1e-5, 10 ** random_state.uniform(-6, -5))
        hparams["proto_domains_per_iter"] = (4, 4)

        # number of rounds of mixup to be done
        hparams["mixup"] = (1.0, 1.0)

        # architecture params
        hparams["bottleneck_size"] = (1024, 1024)

        hparams["batch_size"] = (12, 12)
        # pipeline params and model loading
        # train_prototype tells whether we want to train prototype model or not
        hparams["train_prototype"] = (True, True)
        # proto_model tells path to directory where prototype model is stored
        # prototype model must be called "prototype_final.pth" within this dir
        #hparams["proto_model"] = ("/data/IvLabs/domain_embeddings/outputs/pacs/", "/data/IvLabs/domain_embeddings/outputs/pacs/")
        hparams["proto_model"] = (None, None)
        # fraction of total files to be used to construct prototype
        hparams["proto_train_frac"] = (0.2, 0.2)

        # logging frequency
        hparams["proto_log_train_step"] = (20, 20)
        hparams["proto_log_avg_step"] = (50, 50)

    hparams["resnet_dropout"] = (0.0, float(random_state.choice([0.0, 0.1, 0.5])))

    # TODO clean this up
    hparams.update(
        {
            a: (b, c)
            for a, b, c in [
                # IRM
                ("irm_lambda", 1e2, 10 ** random_state.uniform(-1, 5)),
                (
                    "irm_penalty_anneal_iters",
                    500,
                    int(10 ** random_state.uniform(0, 4)),
                ),
                # Mixup
                ("mixup_alpha", 0.2, 10 ** random_state.uniform(-1, -1)),
                # GroupDRO
                ("groupdro_eta", 1e-2, 10 ** random_state.uniform(-3, -1)),
                # MMD
                ("mmd_gamma", 0.1, 10 ** random_state.uniform(-1, 1)),
                # MLP
                ("mlp_width", 256, int(2 ** random_state.uniform(6, 10))),
                ("mlp_depth", 3, int(random_state.choice([3, 4, 5]))),
                ("mlp_dropout", 0.0, float(random_state.choice([0.0, 0.1, 0.5]))),
                # MLDG
                ("mldg_beta", 1.0, 10 ** random_state.uniform(-1, 1)),
            ]
        }
    )
    return hparams


def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {
        a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()
    }


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}
