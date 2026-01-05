import torch
import numpy as np
import random
from pathlib import Path

class _BaseShardLoader:
    def __init__(self, batch_size, split, data_dir, device):
        self.batch_size = batch_size
        self.split = split
        self.device = device
        
        # Ensure path is a Path object
        self.data_dir = Path(data_dir)
        self.data_root = self.data_dir / split
        
        # Find shards
        self.shards = sorted(list(self.data_root.glob('*.npz')))
        print(f'found {len(self.shards)} shards for split {split}')
        
        # Internal state
        self.remaining_shards = []
        self.current_shard_idx = -1
        self.data_tuple = None
        self.perm = None
        self.current_position = 0
        self.total_samples_in_shard = 0
        
        # Initialize
        self.reset()

    def reset(self):
        self.remaining_shards = list(self.shards)
        random.shuffle(self.remaining_shards)
        self.current_shard_idx = -1
        self.load_next_shard()

    def load_file(self, filename):
        '''Placeholder overwritten by child class'''
        raise NotImplementedError

    def load_next_shard(self):
        self.current_shard_idx += 1
        
        # Check if epoch is done (all shards visited)
        if self.current_shard_idx >= len(self.remaining_shards):
            self.reset()
            return

        filename = self.remaining_shards[self.current_shard_idx]
        
        # Load data using child class implementation
        self.data_tuple = self.load_file(filename)
        
        # Setup shuffling within the shard
        n_samples = len(self.data_tuple[0])
        self.perm = np.random.permutation(n_samples)
        self.current_position = 0
        self.total_samples_in_shard = n_samples

    def next_batch(self):
        # 1. Check if we have enough data left in current shard
        samples_needed = self.batch_size
        remaining_in_shard = self.total_samples_in_shard - self.current_position

        # Case A: The shard has enough data for a full batch
        if remaining_in_shard >= samples_needed:
            indices = self.perm[self.current_position : self.current_position + samples_needed]
            self.current_position += samples_needed
            
            tensors = []
            for arr in self.data_tuple:
                t_data = torch.from_numpy(arr[indices]).to(self.device)
                tensors.append(t_data)
            return tuple(tensors)

        # Case B: The shard does NOT have enough data (Stitching required)
        else:
            # 1. Take everything that is left in the current shard
            indices_part1 = self.perm[self.current_position:]
            
            # Store the partial data (as numpy arrays for now)
            # We loop through the tuple to handle multiple arrays (x, total, etc.)
            partial_batch = [arr[indices_part1] for arr in self.data_tuple]
            
            # 2. Load the next shard
            self.load_next_shard()
            
            # 3. Calculate how much more we need
            samples_still_needed = samples_needed - len(indices_part1)
            
            # 4. Take the remainder from the NEW shard
            indices_part2 = self.perm[0 : samples_still_needed]
            self.current_position = samples_still_needed
            
            # 5. Concatenate and convert
            tensors = []
            # self.data_tuple is now the NEW shard data
            for i, arr_new in enumerate(self.data_tuple):
                part1 = partial_batch[i]
                part2 = arr_new[indices_part2]
                
                # Stack them together
                combined = np.concatenate([part1, part2], axis=0)
                
                t_data = torch.from_numpy(combined).to(self.device)
                tensors.append(t_data)
                
            return tuple(tensors)


class PretrainLoader(_BaseShardLoader):
    def load_file(self, filename):
        print(f'loading {filename}')
        with np.load(filename) as data:
            x = data['x'].astype(np.float32)
            total = data['total'].astype(np.float32)
        return x, total


class TrainingLoader(_BaseShardLoader):
    def load_file(self, filename):
        print(f'loading {filename}')
        with np.load(filename) as data:
            control_x = data['control'].astype(np.float32)
            control_tot = data['control_total'].astype(np.float32)
            case_x = data['case'].astype(np.float32)
            case_tot = data['case_total'].astype(np.float32)
            action_ids = data['action_ids'].astype(np.int64)
        return control_x, control_tot, case_x, case_tot, action_ids

