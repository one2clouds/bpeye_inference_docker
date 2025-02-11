import torch.nn as nn 
import torchvision
import torch
from torchvision.models import ResNet50_Weights

class Res_Net(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(classes))
        
    def forward(self, x:torch.Tensor, num_classes) -> torch.Tensor:
        return self.model(x)