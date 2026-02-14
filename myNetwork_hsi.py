import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkHSI(nn.Module):
    """Feature extractor + classifier wrapper with optional cosine classifier."""

    def __init__(
        self,
        num_classes: int,
        feature_extractor: nn.Module,
        feature_dim: int = 512,
        classifier_type: str = "linear",
        cosine_scale: float = 16.0,
    ):
        super().__init__()
        self.feature = feature_extractor
        self.feature_dim = feature_dim
        self.classifier_type = classifier_type

        if classifier_type == "cosine":
            self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            self.logit_scale = nn.Parameter(torch.tensor(float(cosine_scale), dtype=torch.float32))
            self.fc = None
        else:
            self.fc = nn.Linear(feature_dim, num_classes, bias=True)
            self.weight = None
            self.logit_scale = None

    @property
    def num_classes(self) -> int:
        if self.classifier_type == "cosine":
            return int(self.weight.shape[0])
        return int(self.fc.out_features)

    def classify_from_feature(self, feat: torch.Tensor) -> torch.Tensor:
        if self.classifier_type == "cosine":
            feat_n = F.normalize(feat, p=2, dim=1)
            w_n = F.normalize(self.weight, p=2, dim=1)
            return self.logit_scale * feat_n @ w_n.t()
        return self.fc(feat)

    def forward(self, x):
        feat = self.feature(x)
        logits = self.classify_from_feature(feat)
        return logits, feat

    def incremental_learning(self, num_classes: int):
        if self.classifier_type == "cosine":
            old_weight = self.weight.data.clone()
            new_weight = torch.empty(num_classes, self.feature_dim, device=old_weight.device)
            nn.init.kaiming_uniform_(new_weight, a=5 ** 0.5)
            new_weight[: old_weight.shape[0]] = old_weight
            self.weight = nn.Parameter(new_weight)
            return

        old_weight = self.fc.weight.data.clone()
        old_bias = self.fc.bias.data.clone()
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, num_classes, bias=True)
        self.fc.weight.data[:out_feature] = old_weight
        self.fc.bias.data[:out_feature] = old_bias

    def align_weights(self, old_count: int):
        """Align norms of new classes to old classes to reduce class bias."""
        if old_count <= 0 or old_count >= self.num_classes:
            return

        if self.classifier_type == "cosine":
            return

        with torch.no_grad():
            old_w = self.fc.weight[:old_count]
            new_w = self.fc.weight[old_count:]
            old_norm = old_w.norm(p=2, dim=1).mean()
            new_norm = new_w.norm(p=2, dim=1).mean()
            if new_norm > 0:
                self.fc.weight[old_count:] *= old_norm / new_norm

    def extract_feature(self, x):
        return self.feature(x)
