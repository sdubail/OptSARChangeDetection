import numpy as np
import torch

class WarmupSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, warmup_epochs=2, warmup_ratio=0.8, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio
        self.epoch = 0
        self.rng = np.random.RandomState(seed)
        
        # Extract is_positive values from the metadata
        positive_indices = []
        negative_indices = []
        
        # Use the metadata which is already loaded in memory
        for i, meta in enumerate(dataset.metadata):
            if meta["is_positive"]:
                positive_indices.append(i)
            else:
                negative_indices.append(i)
        
        # Convert to numpy arrays
        self.positive_indices = np.array(positive_indices)
        self.negative_indices = np.array(negative_indices)
        
        print(f"Sampler initialized with {len(self.positive_indices)} positive and "
              f"{len(self.negative_indices)} negative samples")
    
    def __iter__(self):
        if self.epoch < self.warmup_epochs:
            # During warmup: prioritize positive samples
            pos_samples_per_batch = int(self.batch_size * self.warmup_ratio)
            neg_samples_per_batch = self.batch_size - pos_samples_per_batch
            
            # Create batches with the desired ratio
            indices = []
            
            # Calculate number of full batches possible
            num_batches = len(self.dataset) // self.batch_size
            
            for _ in range(num_batches):
                # Use numpy's random choice for efficient sampling
                batch_pos_indices = self.rng.choice(
                    self.positive_indices, 
                    size=pos_samples_per_batch,
                    replace=len(self.positive_indices) < pos_samples_per_batch
                )
                
                batch_neg_indices = self.rng.choice(
                    self.negative_indices, 
                    size=neg_samples_per_batch,
                    replace=False
                )
                
                # Combine and shuffle within batch
                batch_indices = np.concatenate([batch_pos_indices, batch_neg_indices])
                self.rng.shuffle(batch_indices)
                indices.extend(batch_indices.tolist())
                
            # Handle remaining samples if needed
            remaining = len(self.dataset) % self.batch_size
            if remaining > 0:
                remaining_indices = self.rng.choice(len(self.dataset), size=remaining, replace=False)
                indices.extend(remaining_indices.tolist())
            
            return iter(indices)
        else:
            # After warmup: regular sampling
            indices = self.rng.permutation(len(self.dataset)).tolist()
            return iter(indices)
    
    def __len__(self):
        return len(self.dataset)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        # Use a combination of epoch and initial seed to create a new seed
        new_seed = 42 + epoch  # Simple deterministic seed based on epoch
        self.rng = np.random.RandomState(new_seed)