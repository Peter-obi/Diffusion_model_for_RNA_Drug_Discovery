import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.data import PDBRNA, parse_sdf_file
from .pl_data import RNALigandData, torchify_dict


class RNALigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.db = None
        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = [int(key.decode()) for key in txn.cursor().iternext(values=False)]

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (rna_fn, ligand_fn, _) in enumerate(tqdm(index)):
                try:
                    rna_dict = PDBRNA(os.path.join(self.raw_path, rna_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    data = RNALigandData.from_rna_ligand_dicts(
                        rna_dict=torchify_dict(rna_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.rna_filename = rna_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except Exception as e:
                    num_skipped += 1
                    print(f'Skipping ({num_skipped}) {ligand_fn} due to error: {e}')
                    import traceback
                    traceback.print_exc()
                    continue
        db.close()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = str(idx).encode()
        data = pickle.loads(self.db.begin().get(key))
        data = RNALigandData(**data)
        data.id = idx
        assert data.rna_pos.size(0) > 0
        return data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = RNALigandPairDataset(args.path)
    print(len(dataset), dataset[0])
