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
        
    def forward(self, optical_features, sar_features, targets):
        """
        Args:
            optical_features: Features from optical encoder (batch_size, feature_dim)
            sar_features: Features from SAR encoder (batch_size, feature_dim)
            loc_labels: Binary labels (0: no damage, 1: damage) (batch_size,)
        """
        loc_labels = targets["loc_label"]

        # Normalize features
        optical_features = F.normalize(optical_features, dim=1)
        sar_features = F.normalize(sar_features, dim=1)
        
        # Compute cosine similarity for each pair (diagonal of similarity matrix)
        similarity = torch.sum(optical_features * sar_features, dim=1)
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        # For undamaged pairs (label=0): maximize similarity (minimize distance)
        # For damaged pairs (label=1): minimize similarity (maximize distance)
        target_similarity = 1.0 - loc_labels.float()  # 1 for undamaged, 0 for damaged
        
        # Binary cross-entropy loss
        loss = -target_similarity * torch.log(torch.sigmoid(similarity)) - \
               (1 - target_similarity) * torch.log(1 - torch.sigmoid(similarity))
        
        return loss.mean()
    
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