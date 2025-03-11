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
        Args:
            optical_features: Features from optical images (batch_size, feature_dim)
            sar_features: Features from SAR images (batch_size, feature_dim)
            is_positive: Binary labels (0: negative pair with damage, 1: positive pair without damage)
        """
        batch_size = optical_features.shape[0]

        # Normalize features to unit length
        optical_features = F.normalize(optical_features, dim=1)
        sar_features = F.normalize(sar_features, dim=1)

        # Compute full similarity matrix between all pairs in the batch
        # Shape: (batch_size, batch_size)
        similarity_matrix = (
            torch.matmul(optical_features, sar_features.T) / self.temperature
        )

        # Flatten the labels for easier indexing
        labels = is_positive.flatten()

        # Create masks for the positive-negative contrasts:
        # For each positive example i, identify all negative examples j for contrast
        mask_pos_contrast = labels.unsqueeze(1) * (1 - labels).unsqueeze(
            0
        )  # (i,j)=1 if i is positive and j is negative

        # For each negative example i, identify all positive examples j for contrast
        mask_neg_contrast = (1 - labels).unsqueeze(1) * labels.unsqueeze(
            0
        )  # (i,j)=1 if i is negative and j is positive

        # Initialize loss calculation
        loss = torch.tensor(0.0, device=optical_features.device)
        n_comparisons = 0

        # PART 1: For each positive example, contrast with all negative examples
        if torch.sum(mask_pos_contrast) > 0:
            # Get diagonal similarities (corresponding pairs)
            pos_similarities = (
                similarity_matrix.diagonal()
            )  # similarities of pairs (i,i)

            # Get exp(similarity) for positive examples
            pos_exp = torch.exp(pos_similarities[labels == 1])

            # For each positive example, sum exp(similarity) with all negative examples
            # Reshape to get a matrix where each row corresponds to a positive example
            # and contains similarities with all negative examples
            neg_exp_sum = torch.sum(
                torch.exp(similarity_matrix[mask_pos_contrast.bool()]).reshape(
                    torch.sum(labels).int(), -1
                ),
                dim=1,
            )

            # Calculate InfoNCE-style loss for positive examples
            # This maximizes similarity for corresponding positive pairs
            # while minimizing similarity with all negative examples
            loss_pos = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8))
            loss += torch.sum(loss_pos)
            n_comparisons += len(loss_pos)

        # PART 2: For each negative example, contrast with all positive examples (optional)
        if torch.sum(mask_neg_contrast) > 0 and self.use_hard_negatives:
            # Get diagonal similarities (corresponding pairs)
            neg_similarities = (
                similarity_matrix.diagonal()
            )  # similarities of pairs (i,i)

            # For negative pairs, we want to MINIMIZE similarity, so negate the similarity
            neg_exp = torch.exp(-neg_similarities[labels == 0])

            # For each negative example, sum exp(-similarity) with all positive examples
            # The negation inverts the optimization goal
            pos_exp_sum = torch.sum(
                torch.exp(-similarity_matrix[mask_neg_contrast.bool()]).reshape(
                    torch.sum(1 - labels).int(), -1
                ),
                dim=1,
            )

            # Calculate InfoNCE-style loss for negative examples
            # This minimizes similarity for corresponding negative pairs
            # while maximizing similarity with other positive examples
            loss_neg = -torch.log(neg_exp / (neg_exp + pos_exp_sum + 1e-8))
            loss += torch.sum(loss_neg)
            n_comparisons += len(loss_neg)

        # Average the loss over all comparisons
        return loss / max(n_comparisons, 1)


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
