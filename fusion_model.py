import torch
import torch.nn as nn
from vulnerability_model import VulnerabilityClassifier

class HybridFusionModel(nn.Module):
    def __init__(self, embedding_dim=768, ast_feature_dim=15, num_classes=4):
        super(HybridFusionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.ast_feature_dim = ast_feature_dim
        
        # Fusion layer combining both embeddings and AST features
        self.fusion_layer = nn.Linear(embedding_dim + ast_feature_dim, 512)
        self.relu = nn.ReLU()
        
        # Classifier head
        self.classifier = VulnerabilityClassifier(input_dim=512, num_classes=num_classes)

    def forward(self, embeddings, ast_features):
        # Concatenate embeddings and AST features
        fused = torch.cat((embeddings, ast_features), dim=1)
        fused = self.fusion_layer(fused)
        fused = self.relu(fused)
        return self.classifier(fused)
