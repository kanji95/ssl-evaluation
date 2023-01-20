import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from lib.resnet import ResNet18, resnet50, resnet101, resnet18

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=False,
                    use_pretrained=True, logger=None):
    model_ft = None

    if model_name == "resnet101":
        """ Resnet101
        """
        # model_ft = models.resnet101(pretrained=use_pretrained)
        model_ft = resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        """ Resnet50
        """
        # model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft = resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet18":
        """ Resnet18
        """
        # model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft = resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "custom_resnet18":
        model = models.resnet18(pretrained=True)
        model_ft = ResNet18(model, feature_size=600, num_classes=num_classes)
        set_parameter_requires_grad(model_ft, feature_extract)
    return model_ft