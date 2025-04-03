import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class OpticalEncoder(nn.Module):
    """Encoder for optical images with modified ResNet18 or ResNet34 backbone."""

    def __init__(self, resnet_version=18, freeze_resnet=True, in_channels=3):
        """
        Args:
            resnet_version (int): ResNet version to use (18 or 34).
            freeze_resnet (bool): Whether to freeze the ResNet layers before layer4.
            in_channels (int): Number of input channels for the first convolutional layer.
                                Default is 3 for optical images (RGB).
        """
        super(OpticalEncoder, self).__init__()

        # Load pretrained ResNet18 or ResNet34
        if resnet_version == 18:
            print("Loading resnet 18")
            resnet = models.resnet18(pretrained=True)
        elif resnet_version == 34:
            print("Loading resnet 34")
            resnet = models.resnet34(pretrained=True)
        else:
            raise ValueError(f"{resnet_version} - is not a valid resnet version value.")

        # Freeze all layers before layer4 if freeze_resnet is True
        if freeze_resnet:
            for name, param in resnet.named_parameters():
                if not any(layer in name for layer in ["layer4", "fc"]):
                    param.requires_grad = False

        # Modify first conv layer for input channels
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize from pretrained
        # useful as RGB optical images are close to the generalization scope of ResNet
        if in_channels == 3:
            self.conv1.weight.data = resnet.conv1.weight.data
        else:
            # initialize new channels with mean of RGB weights
            self.conv1.weight.data[:, :3, :, :] = resnet.conv1.weight.data
            if in_channels > 3:
                for i in range(3, in_channels):
                    self.conv1.weight.data[:, i : i + 1, :, :] = (
                        resnet.conv1.weight.data.mean(dim=1, keepdim=True)
                    )

        # freeze the first layer if freeze_resnet is True
        if freeze_resnet:
            for param in self.conv1.parameters():
                param.requires_grad = False

        # rRest of the network
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        # modify stride in layer3 and layer4
        self.layer3 = self._modify_layer_stride(resnet.layer3, stride=1)
        self.layer4 = self._modify_layer_stride(resnet.layer4, stride=1)

        self.avgpool = resnet.avgpool
        self.feature_dim = 512

    def _modify_layer_stride(self, layer, stride=1):
        """Modify stride of the first block in a ResNet layer."""
        layer_copy = layer
        if hasattr(layer[0], "conv1"):
            layer_copy[0].conv1.stride = (stride, stride)
        if hasattr(layer[0], "conv2"):
            layer_copy[0].conv2.stride = (stride, stride)
        if hasattr(layer[0], "downsample") and layer[0].downsample is not None:
            layer_copy[0].downsample[0].stride = (stride, stride)
        return layer_copy

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class SAREncoder(nn.Module):
    """Encoder for SAR images with modified ResNet18 or ResNet34 backbone."""

    def __init__(self, resnet_version=18, freeze_resnet=False, in_channels=1):
        """
        Args:
            resnet_version (int): ResNet version to use (18 or 34).
            freeze_resnet (bool): Whether to freeze the ResNet layers before layer4.
                                Default is False for SAR images, because they are not RGB like the training images for ResNet.
            in_channels (int): Number of input channels for the first convolutional layer.
                                Default is 1 for SAR images, because they are grayscale (amplitude).
        """
        super(SAREncoder, self).__init__()

        # Load pretrained ResNet18 or ResNet34
        if resnet_version == 18:
            print("Loading resnet 18")
            resnet = models.resnet18(pretrained=True)
        elif resnet_version == 34:
            print("Loading resnet 34")
            resnet = models.resnet34(pretrained=True)
        else:
            raise ValueError(f"{resnet_version} - is not a valid resnet version value.")

        # Freeze all layers before layer4 if freeze_resnet is True
        if freeze_resnet:
            for name, param in resnet.named_parameters():
                if not any(layer in name for layer in ["layer4", "fc"]):
                    param.requires_grad = False

        # modify first conv layer for input channels
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize from pretrained
        # For SAR images, initialize with mean of RGB channels
        self.conv1.weight.data = resnet.conv1.weight.data.mean(
            dim=1, keepdim=True
        ).repeat(1, in_channels, 1, 1)

        # freeze the first layer if freeze_resnet is True
        if freeze_resnet:
            for param in self.conv1.parameters():
                param.requires_grad = False

        # Rest of the network
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        # modify stride in layer3 and layer4
        self.layer3 = self._modify_layer_stride(resnet.layer3, stride=1)
        self.layer4 = self._modify_layer_stride(resnet.layer4, stride=1)

        self.avgpool = resnet.avgpool
        self.feature_dim = 512

    def _modify_layer_stride(self, layer, stride=1):
        """Modify stride of the first block in a ResNet layer."""
        layer_copy = layer
        if hasattr(layer[0], "conv1"):
            layer_copy[0].conv1.stride = (stride, stride)
        if hasattr(layer[0], "conv2"):
            layer_copy[0].conv2.stride = (stride, stride)
        if hasattr(layer[0], "downsample") and layer[0].downsample is not None:
            layer_copy[0].downsample[0].stride = (stride, stride)
        return layer_copy

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class Projector(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super(Projector, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MultimodalDamageNet(nn.Module):
    """
    Multimodal network for damage assessment with optical pre-event and SAR post-event images.
    Uses contrastive learning.
    """

    def __init__(
        self,
        resnet_version=18,
        freeze_resnet=True,
        optical_channels=3,
        sar_channels=1,
        projection_dim=128,
    ):
        super(MultimodalDamageNet, self).__init__()

        # Encoders
        self.optical_encoder = OpticalEncoder(
            resnet_version=resnet_version,
            freeze_resnet=freeze_resnet,
            in_channels=optical_channels,
        )
        self.sar_encoder = SAREncoder(
            resnet_version=resnet_version,
            freeze_resnet=freeze_resnet,
            in_channels=sar_channels,
        )

        # Feature dimensions
        self.optical_dim = self.optical_encoder.feature_dim
        self.sar_dim = self.sar_encoder.feature_dim

        # Projection heads
        self.optical_projector = Projector(self.optical_dim, out_dim=projection_dim)
        self.sar_projector = Projector(self.sar_dim, out_dim=projection_dim)

    def forward(self, optical=None, sar=None):
        result = {}

        if optical is not None:
            optical_features = self.optical_encoder(optical)
            optical_projected = self.optical_projector(optical_features)
            result["optical_features"] = optical_features
            result["optical_projected"] = optical_projected

            result["pre_features"] = optical_features
            result["pre_projected"] = optical_projected

        if sar is not None:
            sar_features = self.sar_encoder(sar)
            sar_projected = self.sar_projector(sar_features)
            result["sar_features"] = sar_features
            result["sar_projected"] = sar_projected

            result["post_features"] = sar_features
            result["post_projected"] = sar_projected

        if optical is not None and sar is not None:
            # Compute change score
            optical_proj_norm = F.normalize(result["optical_projected"], dim=1)
            sar_proj_norm = F.normalize(result["sar_projected"], dim=1)
            similarity = torch.sum(optical_proj_norm * sar_proj_norm, dim=1)
            result["change_score"] = 1.0 - similarity  # Higher score means more change

        return result

    def save(self, path):
        """
        Save the model's weights.
        Args:
            path (str): path where to save the model
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path, device="cuda"):
        """
        Load the model's weights.
        Args:
            path (str): path to the saved model
            device (str): device where to load the model
        """
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")
