import numpy as np
import torch


class RatioSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, neg_ratio=0.8, tolerance=0.1, seed=42):
        """
        Sampler that enforces a specific ratio of negative to positive samples in each batch.
        Will only resample if the dataset's ratio is outside the tolerance range.

        Args:
            dataset: OnTheFlyNumpyDataset to sample from
            batch_size: Size of each batch
            neg_ratio: Desired ratio of negative samples (0.0-1.0)
            tolerance: Tolerance for ratio difference (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.pos_ratio = 1.0 - neg_ratio
        self.tolerance = tolerance
        self.epoch = 0
        self.rng = np.random.RandomState(seed)

        # Get current dataset ratio
        current_pos_ratio = dataset.get_pos_neg_ratio()
        current_neg_ratio = 1.0 - current_pos_ratio

        # Determine if we need to enforce ratio
        self.enforce_ratio = abs(current_neg_ratio - neg_ratio) > tolerance

        self.positive_indices = dataset.selected_pos_indices
        self.negative_indices = dataset.selected_neg_indices

        # Compute statistics
        self.n_pos = len(self.positive_indices)
        self.n_neg = len(self.negative_indices)
        self.dataset_pos_ratio = self.n_pos / (self.n_pos + self.n_neg)
        self.dataset_neg_ratio = 1.0 - self.dataset_pos_ratio

        print("Dataset statistics:")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Positive samples: {self.n_pos} ({self.dataset_pos_ratio:.2%})")
        print(f"  - Negative samples: {self.n_neg} ({self.dataset_neg_ratio:.2%})")
        print(
            f"Target sampling ratio - Negative: {self.neg_ratio:.2%}, Positive: {self.pos_ratio:.2%}"
        )

        if self.enforce_ratio:
            print(
                f"Current ratio differs from target by more than {tolerance:.1%}, will enforce target ratio"
            )

            # Calculate how many samples of each class we need per batch
            self.neg_per_batch = int(self.batch_size * self.neg_ratio)
            self.pos_per_batch = self.batch_size - self.neg_per_batch

            print(
                f"Each batch will contain {self.neg_per_batch} negative and {self.pos_per_batch} positive samples"
            )

            # Calculate oversampling statistics
            total_batches = len(dataset) // self.batch_size + (
                1 if len(dataset) % self.batch_size > 0 else 0
            )
            total_neg_needed = self.neg_per_batch * total_batches
            neg_oversampling_factor = total_neg_needed / self.n_neg
            if neg_oversampling_factor > 1:
                print(
                    f"Negative samples will be oversampled by approximately {neg_oversampling_factor:.2f}x"
                )
        else:
            print(
                f"Current ratio {self.dataset_neg_ratio:.2%} is within tolerance of target {self.neg_ratio:.2%}, using regular shuffling"
            )

    def __iter__(self):
        """Generate batches with or without enforced ratio."""
        if not self.enforce_ratio:
            # If dataset already has the right ratio, just shuffle
            indices = np.concatenate([self.positive_indices, self.negative_indices])
            self.rng.shuffle(indices)
            return iter(indices)

        # Otherwise, enforce ratio per batch
        # Calculate number of batches
        n_batches = len(self.dataset) // self.batch_size
        if len(self.dataset) % self.batch_size > 0:
            n_batches += 1

        # Shuffle all indices at the start of each iteration
        positive_indices = self.positive_indices.copy()
        negative_indices = self.negative_indices.copy()
        self.rng.shuffle(positive_indices)
        self.rng.shuffle(negative_indices)

        # Create circular iterators to ensure we never run out of samples
        pos_iterator = self._circular_iterator(positive_indices)
        neg_iterator = self._circular_iterator(negative_indices)

        all_indices = []
        for _ in range(n_batches):
            # Sample the required number of each class
            batch_pos_indices = [next(pos_iterator) for _ in range(self.pos_per_batch)]
            batch_neg_indices = [next(neg_iterator) for _ in range(self.neg_per_batch)]

            # Combine and shuffle
            batch_indices = batch_pos_indices + batch_neg_indices
            self.rng.shuffle(batch_indices)

            all_indices.extend(batch_indices)

        return iter(all_indices)

    def _circular_iterator(self, indices):
        """Create an iterator that cycles through indices forever, with reshuffling."""
        while True:
            self.rng.shuffle(indices)  # Shuffle before each pass
            for idx in indices:
                yield idx

    def __len__(self):
        """Return the total dataset size."""
        return len(self.dataset)

    def set_epoch(self, epoch):
        """Update the random seed for the new epoch."""
        self.epoch = epoch
        self.rng = np.random.RandomState(42 + epoch)


#####################################################
################### OLDER VERSION ###################
##################################################### 
#### A warmup sampler that gradually increases the ratio of positive samples in the batches.
#### This is useful for training models that need to learn from a small number of negative samples first.

# class WarmupSampler(torch.utils.data.Sampler):
#     def __init__(self, dataset, batch_size, warmup_epochs=2, seed=42):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.warmup_epochs = warmup_epochs
#         self.epoch = 0
#         self.rng = np.random.RandomState(seed)

#         # Extract positive and negative indices
#         self.positive_indices = np.array(
#             [i for i, meta in enumerate(dataset.metadata) if meta["is_positive"]]
#         )
#         self.negative_indices = np.array(
#             [i for i, meta in enumerate(dataset.metadata) if not meta["is_positive"]]
#         )

#         # Compute the true positive ratio in the dataset
#         self.warmup_ratio = len(self.positive_indices) / len(self.dataset)
#         print(f"Warmup ratio: {self.warmup_ratio:.4f}")

#     def __iter__(self):
#         if self.epoch < self.warmup_epochs:
#             # Adjust the warmup ratio based on available data
#             pos_per_batch = int(self.batch_size * self.warmup_ratio)
#             neg_per_batch = self.batch_size - pos_per_batch

#             num_batches = len(self.dataset) // self.batch_size
#             batches = []

#             for _ in range(num_batches):
#                 batch_pos = self.rng.choice(
#                     self.positive_indices, size=pos_per_batch, replace=False
#                 ).tolist()
#                 batch_neg = self.rng.choice(
#                     self.negative_indices, size=neg_per_batch, replace=False
#                 ).tolist()
#                 batches.append(batch_pos + batch_neg)

#             indices = [idx for batch in batches for idx in batch]

#             # Handle remaining samples
#             remaining = len(self.dataset) % self.batch_size
#             if remaining > 0:
#                 remaining_indices = self.rng.choice(
#                     len(self.dataset), size=remaining, replace=False
#                 ).tolist()
#                 indices.extend(remaining_indices)

#             return iter(indices)
#         else:
#             # After warmup: Fully shuffle dataset
#             indices = self.rng.permutation(len(self.dataset)).tolist()
#             return iter(indices)

#     def __len__(self):
#         return len(self.dataset)

#     def set_epoch(self, epoch):
#         self.epoch = epoch
#         self.rng = np.random.RandomState(
#             42 + epoch
#         )  # Update RNG with deterministic seed