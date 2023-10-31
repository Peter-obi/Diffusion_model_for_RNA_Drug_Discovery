import argparse
import random
import torch
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./processed_data')
    parser.add_argument('--dest', type=str, default='./training_data/crossdocked_pocket10_pose_split.pt')
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    # Assuming each file in the directory is a PDB file
    all_files = [f for f in os.listdir(args.path) if f.endswith('.pdb')]
    
    print('Found files:', all_files)

    # Ensure we have exactly 6 PDB files
    assert len(all_files) == 6, "Expected exactly 6 PDB files in the directory."

    # Shuffle the file names for randomness
    random.Random(args.seed).shuffle(all_files)

    # Assign 4 for training, 1 for validation, and 1 for testing
    train_files = all_files[:4]
    val_files = [all_files[4]]
    test_files = [all_files[5]]

    # Save the filenames
    torch.save({
        'train': train_files,
        'val': val_files,
        'test': test_files,
    }, args.dest)

    print('Train Files:', train_files)
    print('Validation Files:', val_files)
    print('Test Files:', test_files)
    print('Done.')

