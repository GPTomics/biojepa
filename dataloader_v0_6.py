import torch
import numpy as np
import random
from pathlib import Path
from math import ceil
from collections import defaultdict, namedtuple

TrainingBatch = namedtuple('TrainingBatch', [
    'control', 'control_total', 'case', 'case_total',
    'seq_idx', 'target_idx', 'modality', 'mode',
    'has_seq', 'has_target', 'n_perts', 'dose',
    'batch_id', 'cell_type'
])

AlignmentBatch = namedtuple('AlignmentBatch', ['seq_idx', 'target_idx', 'modality', 'mode'])

PretrainBatch = namedtuple('PretrainBatch', ['x', 'total'])


def _parse_dataset_name(shard_path, split):
    parts = shard_path.stem.split('_')
    if split not in parts:
        return None
    split_idx = parts.index(split)
    if split_idx < 2:
        return None
    return '_'.join(parts[1:split_idx])


class _BaseShardLoader:
    def __init__(self, batch_size, split, data_dir, device, total_samples=None, min_dataset_fraction=0.1):
        self.batch_size = batch_size
        self.split = split
        self.device = device
        self.min_dataset_fraction = min_dataset_fraction

        self.data_dir = Path(data_dir)
        self.data_root = self.data_dir / split

        self.shards = sorted(list(self.data_root.glob('*.npz')))
        print(f'found {len(self.shards)} shards for split {split}')
        self._balance_shards()

        self.remaining_shards = []
        self.current_shard_idx = -1
        self.data_tuple = None
        self.perm = None
        self.current_position = 0
        self.total_samples_in_shard = 0

        self.reset()

        if total_samples is not None:
            self.total_samples = total_samples
        else:
            self.total_samples = self.total_samples_in_shard * len(self.shards)

    def _balance_shards(self):
        if self.split != 'train':
            return
        shards_by_dataset = defaultdict(list)
        unparsed = []
        for shard in self.shards:
            parts = shard.stem.split('_')
            if self.split not in parts:
                unparsed.append(shard)
                continue
            split_idx = parts.index(self.split)
            if split_idx < 2:
                unparsed.append(shard)
                continue
            dataset = '_'.join(parts[1:split_idx])
            shards_by_dataset[dataset].append(shard)
        if len(shards_by_dataset) <= 1:
            return
        max_count = max(len(s) for s in shards_by_dataset.values())
        threshold = ceil(max_count * self.min_dataset_fraction)
        balanced = list(unparsed)
        print(f'shard balancing (threshold={threshold}, {self.min_dataset_fraction:.0%} of max={max_count}):')
        for dataset in sorted(shards_by_dataset):
            original = shards_by_dataset[dataset]
            count = len(original)
            if count >= threshold:
                balanced.extend(original)
                print(f'  {dataset}: {count} shards')
            else:
                multiplier = ceil(threshold / count)
                balanced.extend(original * multiplier)
                print(f'  {dataset}: {count} -> {count * multiplier} shards (x{multiplier})')
        self.shards = sorted(balanced)

    def reset(self):
        self.remaining_shards = list(self.shards)
        random.shuffle(self.remaining_shards)
        self.current_shard_idx = -1
        self.load_next_shard()

    def load_file(self, filename):
        raise NotImplementedError

    def load_next_shard(self):
        if not self.remaining_shards:
            raise RuntimeError(f'No shards found in {self.data_root}')

        self.current_shard_idx += 1

        if self.current_shard_idx >= len(self.remaining_shards):
            self.reset()
            return

        filename = self.remaining_shards[self.current_shard_idx]
        self.data_tuple = self.load_file(filename)

        n_samples = len(self.data_tuple[0])
        self.perm = np.random.permutation(n_samples)
        self.current_position = 0
        self.total_samples_in_shard = n_samples

    def next_batch(self):
        samples_needed = self.batch_size
        remaining_in_shard = self.total_samples_in_shard - self.current_position

        if remaining_in_shard >= samples_needed:
            indices = self.perm[self.current_position : self.current_position + samples_needed]
            self.current_position += samples_needed

            tensors = []
            for arr in self.data_tuple:
                dtype = torch.float32 if arr.dtype.kind == 'f' else torch.long
                if arr.dtype == np.bool_:
                    dtype = torch.bool
                t_data = torch.from_numpy(arr[indices]).to(dtype=dtype, device=self.device)
                tensors.append(t_data)
            return tuple(tensors)

        else:
            indices_part1 = self.perm[self.current_position:]
            partial_batch = [arr[indices_part1] for arr in self.data_tuple]

            self.load_next_shard()

            samples_still_needed = samples_needed - len(indices_part1)
            indices_part2 = self.perm[0 : samples_still_needed]
            self.current_position = samples_still_needed

            tensors = []
            for i, arr_new in enumerate(self.data_tuple):
                part1 = partial_batch[i]
                part2 = arr_new[indices_part2]
                combined = np.concatenate([part1, part2], axis=0)

                dtype = torch.float32 if combined.dtype.kind == 'f' else torch.long
                if combined.dtype == np.bool_:
                    dtype = torch.bool
                t_data = torch.from_numpy(combined).to(dtype=dtype, device=self.device)
                tensors.append(t_data)

            return tuple(tensors)


class PretrainLoader(_BaseShardLoader):
    def __init__(self, batch_size, split, data_dir, device, total_samples=None):
        super().__init__(batch_size, split, data_dir, device, total_samples)

    def load_file(self, filename):
        #print(f'loading {filename}')
        with np.load(filename) as data:
            x = data['x'].astype(np.float32)
            total = data['total'].astype(np.float32)
        return x, total

    def next_batch(self):
        batch = super().next_batch()
        return PretrainBatch(*batch)


class AlignmentLoader(_BaseShardLoader):
    def __init__(self, batch_size, split, data_dir, device, total_samples=None):
        super().__init__(batch_size, split, data_dir, device, total_samples)

    def load_file(self, filename):
        #print(f'loading {filename}')
        with np.load(filename) as data:
            seq_idx = data['seq_idx'].astype(np.int64)
            target_idx = data['target_idx'].astype(np.int64)
            modality = data['modality'].astype(np.int64)
            mode = data['mode'].astype(np.int64)
        return seq_idx, target_idx, modality, mode

    def next_batch(self):
        batch = super().next_batch()
        return AlignmentBatch(*batch)


class TrainingLoader(_BaseShardLoader):
    def __init__(self, batch_size, split, data_dir, device, total_samples=None):
        self._dataset_offsets = None
        super().__init__(batch_size, split, data_dir, device, total_samples)

    def _build_dataset_offsets(self):
        names = set()
        for shard in self.shards:
            name = _parse_dataset_name(shard, self.split)
            if name:
                names.add(name)
        name_to_offset = {name: i * 10000 for i, name in enumerate(sorted(names))}
        self._dataset_offsets = {}
        for shard in self.shards:
            name = _parse_dataset_name(shard, self.split)
            self._dataset_offsets[shard] = name_to_offset.get(name, 0)

    def load_file(self, filename):
        if self._dataset_offsets is None:
            self._build_dataset_offsets()
        with np.load(filename) as data:
            control_x = data['control'].astype(np.float32)
            control_tot = data['control_total'].astype(np.float32)
            case_x = data['case'].astype(np.float32)
            case_tot = data['case_total'].astype(np.float32)

            seq_idx = data['seq_idx'].astype(np.int64)
            target_idx = data['target_idx'].astype(np.int64)
            modality = data['modality'].astype(np.int64)
            mode = data['mode'].astype(np.int64)
            has_seq = data['has_seq'].astype(np.bool_)
            has_target = data['has_target'].astype(np.bool_)
            n_perts = data['n_perts'].astype(np.int64)
            dose = data['dose'].astype(np.float32)

            batch_id = data['batch_id'].astype(np.int64) if 'batch_id' in data else np.zeros(len(control_x), dtype=np.int64)
            cell_type = data['cell_type'].astype(np.int64) if 'cell_type' in data else np.zeros(len(control_x), dtype=np.int64)

        batch_id = batch_id + self._dataset_offsets.get(filename, 0)

        return (control_x, control_tot, case_x, case_tot, seq_idx, target_idx,
                modality, mode, has_seq, has_target, n_perts, dose, batch_id, cell_type)

    def next_batch(self):
        batch = super().next_batch()
        return TrainingBatch(*batch)
