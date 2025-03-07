import numpy as np
import torch

class WarmupSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, warmup_epochs=2, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.rng = np.random.RandomState(seed)

        # Extract positive and negative indices
        self.positive_indices = np.array([i for i, meta in enumerate(dataset.metadata) if meta["is_positive"]])
        self.negative_indices = np.array([i for i, meta in enumerate(dataset.metadata) if not meta["is_positive"]])

        # Compute the true positive ratio in the dataset
        self.warmup_ratio = len(self.positive_indices) / len(self.dataset)
        print(f"Warmup ratio: {self.warmup_ratio:.4f}")

    def __iter__(self):
        if self.epoch < self.warmup_epochs:
            # Adjust the warmup ratio based on available data
            pos_per_batch = int(self.batch_size * self.warmup_ratio)
            neg_per_batch = self.batch_size - pos_per_batch

            num_batches = len(self.dataset) // self.batch_size
            batches = []

            for _ in range(num_batches):
                batch_pos = self.rng.choice(self.positive_indices, size=pos_per_batch, replace=False).tolist()
                batch_neg = self.rng.choice(self.negative_indices, size=neg_per_batch, replace=False).tolist()
                batches.append(batch_pos + batch_neg)

            indices = [idx for batch in batches for idx in batch]

            # Handle remaining samples
            remaining = len(self.dataset) % self.batch_size
            if remaining > 0:
                remaining_indices = self.rng.choice(len(self.dataset), size=remaining, replace=False).tolist()
                indices.extend(remaining_indices)

            return iter(indices)
        else:
            # After warmup: Fully shuffle dataset
            indices = self.rng.permutation(len(self.dataset)).tolist()
            return iter(indices)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = np.random.RandomState(42 + epoch)  # Update RNG with deterministic seed
