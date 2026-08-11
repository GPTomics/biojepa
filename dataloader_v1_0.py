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
    'batch_id', 'cell_type', 'gene_mask'
])

EvalBatch = namedtuple('EvalBatch', TrainingBatch._fields + ('dataset_id',))

ComposerBatch = namedtuple('ComposerBatch', ['seq_idx', 'target_idx', 'modality', 'mode'])
AlignmentBatch = ComposerBatch

EncoderBatch = namedtuple('EncoderBatch', ['x', 'total', 'gene_mask'])
PretrainBatch = EncoderBatch


def _parse_dataset_name(shard_path, split):
    parts = shard_path.stem.split('_')
    if split not in parts:
        return None
    split_idx = parts.index(split)
    if split_idx < 2:
        return None
    return '_'.join(parts[1:split_idx])


class _BaseShardLoader:
    sample_count_key = None

    def __init__(self, batch_size, split, data_dir, device, total_samples=None, min_dataset_fraction=0.2, seed=None):
        self.batch_size = batch_size
        self.split = split
        self.device = device
        self.min_dataset_fraction = min_dataset_fraction
        self._rng_py = random.Random(seed) if seed is not None else random.Random()
        self._rng_np = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

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
            self.total_samples = self._calculate_total_samples()

    def _count_shard_samples(self, shard_path):
        with np.load(shard_path) as data:
            return data[self.sample_count_key].shape[0]

    def _calculate_total_samples(self):
        counts = {
            shard: self._count_shard_samples(shard)
            for shard in dict.fromkeys(self.shards)
        }
        return sum(counts[shard] for shard in self.shards)

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
        self._rng_py.shuffle(self.remaining_shards)
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
        self.perm = self._rng_np.permutation(n_samples)
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


class EncoderLoader(_BaseShardLoader):
    sample_count_key = 'total'

    def __init__(self, batch_size, split, data_dir, device, total_samples=None, seed=None):
        super().__init__(batch_size, split, data_dir, device, total_samples, seed=seed)

    def load_file(self, filename):
        with np.load(filename) as data:
            x = data['x'].astype(np.float32)
            total = data['total'].astype(np.float32)
            if 'gene_mask' in data:
                gene_mask = data['gene_mask'].astype(np.bool_)
                gene_mask = np.broadcast_to(gene_mask, (len(x), len(gene_mask))).copy()
            else:
                gene_mask = np.ones(x.shape, dtype=np.bool_)
        return x, total, gene_mask

    def next_batch(self):
        batch = super().next_batch()
        return EncoderBatch(*batch)

PretrainLoader = EncoderLoader


class ComposerLoader(_BaseShardLoader):
    sample_count_key = 'modality'

    def __init__(self, batch_size, split, data_dir, device, total_samples=None, seed=None, chemical_fraction=0.0):
        self.chemical_fraction = chemical_fraction
        super().__init__(batch_size, split, data_dir, device, total_samples, seed=seed)
        if chemical_fraction > 0 and split == 'train':
            print(f'  modality balancing: target chemical_fraction={chemical_fraction:.0%}')
            if total_samples is None:
                self.total_samples = self._count_total_with_balancing()
                print(f'  adjusted total_samples={self.total_samples} (from {len(self.shards)} shards)')

    def _count_total_with_balancing(self):
        total = 0
        for shard_path in self.shards:
            with np.load(shard_path) as data:
                modality = data['modality']
            n_total = len(modality)
            chem_mask = modality == 2
            n_chem = int(chem_mask.sum())
            if n_chem > 0 and n_chem / n_total < self.chemical_fraction:
                target_chem = int(n_total * self.chemical_fraction / (1 - self.chemical_fraction))
                multiplier = ceil(target_chem / n_chem)
                if multiplier > 1:
                    extra = min((multiplier - 1) * n_chem, target_chem - n_chem)
                    total += n_total + extra
                else:
                    total += n_total
            else:
                total += n_total
        return total

    def load_file(self, filename):
        with np.load(filename) as data:
            seq_idx = data['seq_idx'].astype(np.int64)
            target_idx = data['target_idx'].astype(np.int64)
            modality = data['modality'].astype(np.int64)
            mode = data['mode'].astype(np.int64)

        if self.chemical_fraction > 0 and self.split == 'train':
            chem_mask = modality == 2
            n_chem = int(chem_mask.sum())
            n_total = len(modality)
            if n_chem > 0 and n_chem / n_total < self.chemical_fraction:
                target_chem = int(n_total * self.chemical_fraction / (1 - self.chemical_fraction))
                multiplier = ceil(target_chem / n_chem)
                if multiplier > 1:
                    chem_indices = np.where(chem_mask)[0]
                    extra = np.tile(chem_indices, multiplier - 1)[:target_chem - n_chem]
                    all_idx = np.concatenate([np.arange(n_total), extra])
                    seq_idx, target_idx = seq_idx[all_idx], target_idx[all_idx]
                    modality, mode = modality[all_idx], mode[all_idx]

        return seq_idx, target_idx, modality, mode

    def next_batch(self):
        batch = super().next_batch()
        return ComposerBatch(*batch)

AlignmentLoader = ComposerLoader


class TrainingLoader(_BaseShardLoader):
    sample_count_key = 'control_total'

    def __init__(self, batch_size, split, data_dir, device, total_samples=None, seed=None):
        self._dataset_offsets = None
        super().__init__(batch_size, split, data_dir, device, total_samples, seed=seed)

    def _build_dataset_offsets(self):
        names = set()
        for shard in self.shards:
            name = _parse_dataset_name(shard, self.split)
            if name:
                names.add(name)
        self._dataset_names = sorted(names)
        name_to_offset = {name: i * 10000 for i, name in enumerate(self._dataset_names)}
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
            valid_dose = dose != -1.0
            dose = np.where(valid_dose, np.log1p(np.maximum(dose, 0.0)), dose)

            batch_id = data['batch_id'].astype(np.int64) if 'batch_id' in data else np.zeros(len(control_x), dtype=np.int64)
            cell_type = data['cell_type'].astype(np.int64) if 'cell_type' in data else np.zeros(len(control_x), dtype=np.int64)

            if 'gene_mask' in data:
                gene_mask = data['gene_mask'].astype(np.bool_)
                gene_mask = np.broadcast_to(gene_mask, (len(control_x), len(gene_mask))).copy()
            else:
                gene_mask = np.ones(control_x.shape, dtype=np.bool_)

        batch_id = batch_id + self._dataset_offsets.get(filename, 0)

        return (control_x, control_tot, case_x, case_tot, seq_idx, target_idx,
                modality, mode, has_seq, has_target, n_perts, dose, batch_id, cell_type, gene_mask)

    def next_batch(self):
        batch = super().next_batch()
        return TrainingBatch(*batch)


class EvalLoader(TrainingLoader):

    def __init__(self, batch_size, split, data_dir, device, total_samples=None, seed=None):
        self._dataset_name_to_id = None
        self.dataset_id_to_name = None
        super().__init__(batch_size, split, data_dir, device, total_samples, seed=seed)

    def _build_dataset_offsets(self):
        super()._build_dataset_offsets()
        self._dataset_name_to_id = {name: i for i, name in enumerate(self._dataset_names)}
        self.dataset_id_to_name = {i: name for i, name in enumerate(self._dataset_names)}

    def load_file(self, filename):
        result = super().load_file(filename)
        ds_name = _parse_dataset_name(filename, self.split) or 'unknown'
        ds_id = self._dataset_name_to_id.get(ds_name, -1)
        dataset_ids = np.full(len(result[0]), ds_id, dtype=np.int64)
        return result + (dataset_ids,)

    def next_batch(self):
        batch = _BaseShardLoader.next_batch(self)
        return EvalBatch(*batch)
