import os
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import torchvision.transforms as T
from common.vision.transforms import ResizeImage
import common.vision.datasets.openset as datasets
from common.vision.datasets.openset import default_open_set as open_set
from common.vision.datasets.Concatenate import ConcatenateDataset
from common.utils.data import ForeverDataIterator

def setup_datasets(args, concat=True, return_domain_idx=True):

    # if concat:
    ConcatD = ConcatenateDataset if return_domain_idx else ConcatDataset

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

    val_transform = T.Compose([ResizeImage(256), T.CenterCrop(224), T.ToTensor(), normalize])

    dataset = datasets.__dict__[args.dataset]
    source_dataset = open_set(dataset, source=True)
    target_dataset = open_set(dataset, source=False)
    if type(args.source) == str: # added this if to avoid repeatead modification of args.source
        args.source = args.source.split(",")
        args.target = args.target.split(",")
        if args.dataset == "OfficeHome":
            args.root = os.path.join(args.root, "office-home")

    source_datasets = [
        source_dataset(root=args.root, task=source, download=True, transform=train_transform)
        for source in args.source
    ]
    num_classes = len(source_datasets[0].classes)

    target_datasets = [
        target_dataset(root=args.root, task=target, download=True, transform=train_transform)
        for target in args.target
    ]   

    if concat:
        train_source_dataset = ConcatD(source_datasets)
        train_target_dataset = ConcatD(target_datasets)
    else:
        train_source_dataset = source_datasets
        train_target_dataset = target_datasets

    # Getting Loaders
    if concat:
        train_source_loader = DataLoader(
            train_source_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
        )
        train_target_loader = DataLoader(
            train_target_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
        )
    else:
        train_source_loader = [(DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
        ))
        for dataset in train_source_dataset]
        train_target_loader = [(DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
        ))
        for dataset in train_target_dataset]
        

    # val_dataset = target_dataset(
    #     root=args.root, task=args.target, download=True, transform=val_transform
    # )
    val_dataset = ConcatD([
            target_dataset(root=args.root, task=target, download=True, transform=val_transform)
            for target in args.target
        ]) 

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    if args.dataset == "DomainNet":
        test_dataset = dataset(
            root=args.root,
            task=args.target,
            split="test",
            download=True,
            transform=val_transform,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
    else:
        test_loader = val_loader

    # train_source_iter = ForeverDataIterator(train_source_loader)
    # train_target_iter = ForeverDataIterator(train_target_loader)
    # return num_classes, train_source_dataset, train_target_loader
    return num_classes, train_source_loader, train_target_loader, val_loader, test_loader
    