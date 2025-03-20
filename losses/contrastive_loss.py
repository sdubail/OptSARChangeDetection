import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for multimodal damage assessment.
    """

    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, optical_features, sar_features, is_positive):
        """
        Args:
            optical_features: Features from optical encoder (batch_size, feature_dim)
            sar_features: Features from SAR encoder (batch_size, feature_dim)
            is_positive: Binary labels (0: damage = negative pair, 1: no damage = positive pair) (batch_size,)
        """
        # Normalize features
        optical_features = F.normalize(optical_features, dim=1)
        sar_features = F.normalize(sar_features, dim=1)

        # Compute cosine similarity for each pair (diagonal of similarity matrix)
        similarity = torch.sum(optical_features * sar_features, dim=1)

        # Scale by temperature
        similarity = similarity / self.temperature

        # For undamaged pairs (label=1 - positive pair): maximize similarity (minimize distance)
        # For damaged pairs (label=0 - negative pair): minimize similarity (maximize distance)
        target_similarity = is_positive

        # Binary cross-entropy loss
        loss = -target_similarity * torch.log(torch.sigmoid(similarity)) - (
            1 - target_similarity
        ) * torch.log(1 - torch.sigmoid(similarity))

        return loss.mean()


class InfoNCEContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss implementation that leverages negative samples within the batch.

    This loss pushes corresponding optical-SAR pairs without damage (positive pairs) to be similar,
    while pushing optical features of undamaged buildings away from SAR features of damaged buildings.
    """

    def __init__(self, temperature=0.07, use_hard_negatives=True):
        """
        Args:
            temperature: Scaling factor for the similarity scores (lower = harder contrasts)
            use_hard_negatives: Whether to include the negative pair contrast term in the loss
        """
        super(InfoNCEContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives

    def forward(self, optical_features, sar_features, is_positive):
        """
        Forward pass for the contrastive loss calculation.

        Args:
            optical_features: Features from optical images (batch_size, feature_dim)
            sar_features: Features from SAR images (batch_size, feature_dim)
            is_positive: Binary labels (0: damaged pair = negative, 1: intact pair = positive)

        Returns:
            Scalar loss value (average contrastive loss across all valid comparisons)
        """
        batch_size = optical_features.shape[0]

        # Ensure is_positive is a float tensor for calculations
        if not isinstance(is_positive, torch.Tensor):
            is_positive = torch.tensor(is_positive, device=optical_features.device)
        is_positive = is_positive.float().view(-1)

        # Normalize features to unit vectors
        optical_features = F.normalize(optical_features, dim=1)
        sar_features = F.normalize(sar_features, dim=1)

        # Compute similarity matrix between all pairs in the batch
        similarity_matrix = (
            torch.matmul(optical_features, sar_features.T) / self.temperature
        )

        # Get diagonal similarities (corresponding pairs)
        pair_similarities = similarity_matrix.diagonal()

        # Initialize loss with a tensor that supports gradient calculation
        neg_loss_tot = torch.zeros(
            1, device=optical_features.device, requires_grad=True
        )
        pos_loss_tot = torch.zeros(
            1, device=optical_features.device, requires_grad=True
        )
        n_comparisons = 0

        # Create masks for positive and negative examples
        pos_mask = is_positive == 1
        neg_mask = is_positive == 0

        # PART 1: Process positive examples (intact buildings in both modalities)
        n_positives = pos_mask.sum().item()
        if n_positives > 0:
            pos_similarities = pair_similarities[pos_mask]

            # Create mask for positive-negative contrasts
            # Each row i corresponds to a positive example, each column j to a negative example
            mask_pos_contrast = pos_mask.unsqueeze(1) * neg_mask.unsqueeze(0)

            # If we have both positive and negative examples for contrast
            if mask_pos_contrast.sum() > 0:
                pos_exp = torch.exp(pos_similarities)

                # Reshape to get similarities with negative examples for each positive example
                neg_similarities = similarity_matrix[mask_pos_contrast.bool()].reshape(
                    n_positives, -1
                )
                neg_exp_sum = torch.sum(torch.exp(neg_similarities), dim=1)

                # Calculate InfoNCE-style loss (bring positives closer, push negatives away)
                pos_loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8))
                pos_loss_tot = pos_loss_tot + pos_loss.sum()
                n_comparisons += n_positives
            else:
                # If no negative examples, simply maximize similarity for positive pairs
                pos_loss = -pos_similarities
                pos_loss_tot = pos_loss_tot + pos_loss.sum()
                n_comparisons += n_positives

        # PART 2: Process negative examples (damaged buildings)
        n_negatives = neg_mask.sum().item()
        if n_negatives > 0:
            neg_similarities = pair_similarities[neg_mask]

            # Minimize similarity for negative pairs (push damaged pairs apart)
            neg_loss = torch.log(1 + torch.exp(neg_similarities))
            neg_loss_tot = neg_loss_tot + neg_loss.sum()
            n_comparisons += n_negatives

        # Handle the case where no comparisons are made (empty batch or all one class)
        if n_comparisons == 0:
            return torch.zeros(1, requires_grad=True, device=optical_features.device)

        # Return average loss
        # return pos_loss_tot / n_comparisons, neg_loss_tot / n_comparisons
        return pos_loss_tot / n_positives, neg_loss_tot / n_negatives


# class CombinedLoss(nn.Module):
#     """
#     Combined loss function for multimodal damage assessment.
#     Includes supervised contrastive loss and classification loss.
#     """

#     def __init__(self, contrastive_weight=1.0, temperature=0.07):
#         super(CombinedLoss, self).__init__()
#         self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)
#         self.classification_loss = nn.CrossEntropyLoss()
#         self.contrastive_weight = contrastive_weight

#     def forward(self, outputs, targets):
#         """
#         Args:
#             outputs: Dictionary of model outputs
#             targets: Dictionary of targets

#         Returns:
#             Combined loss
#         """
#         # Contrastive loss
#         contrast_loss = self.contrastive_loss(
#             outputs['optical_projected'],
#             outputs['sar_projected'],
#             targets['loc_label']
#         )

#         # Classification loss
#         class_loss = self.classification_loss(
#             outputs['damage_logits'],
#             targets['label']
#         )

#         # Combined loss
#         loss = class_loss + self.contrastive_weight * contrast_loss

#         return loss, {
#             'contrastive_loss': contrast_loss.item(),
#             'classification_loss': class_loss.item()
#         }
