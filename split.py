import os
import argparse
import random
import torch
from tqdm.auto import tqdm

from torch.utils.data import Subset
from datasets.pl_pair_dataset import RNALigandPairDataset  # Assuming you have a similar class for RNA-ligand pairs


def get_pdb_name(fn):
    return os.path.basename(fn)[:4]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./processed_data/')
    parser.add_argument('--dest', type=str, default='./data/rna_ligand_split.pt')
    parser.add_argument('--train', type=int, default=4)
    parser.add_argument('--val', type=int, default=1)
    parser.add_argument('--test', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    dataset = RNALigandPairDataset(args.path)
    print('Load dataset successfully!')
    print('Number of items in the dataset:', len(dataset))
    if len(dataset) > 0:
       print('Sample data item:', dataset[0])

    all_id = list(range(len(dataset)))
    random.Random(args.seed).shuffle(all_id)

    train_id = all_id[:args.train]
    val_id = all_id[args.train: args.train + args.val]
    test_id = all_id[args.train + args.val: args.train + args.val + args.test]

    torch.save({
        'train': train_id,
        'val': val_id,
        'test': test_id,
    }, args.dest)

    print('Train %d, Validation %d, Test %d.' % (len(train_id), len(val_id), len(test_id)))
    print('Done.')

