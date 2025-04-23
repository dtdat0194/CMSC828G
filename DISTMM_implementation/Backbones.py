'''
Coding assignment from CMSC848M-0101: Selected Topics in Information Processing; Multimodal Computer Vision
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchsummary import summary#check info of the model


class VisionClassifier(nn.Module):
    """Vision-based material classifier
    
    Instructions:
    1. Support both ResNet and FENet backbones
    2. Add classification head with dropout (p=0.5)
    3. Return predictions and loss in a dictionary
    4. Handle both training and inference modes
    """
    def __init__(self, num_classes, backbone='resnet'):
        super().__init__()
        # TODO: Initialize backbone and classifier
        if backbone == "resnet":
            print("vision using resnet")
            resnet18 = models.resnet18(pretrained = True)
            first_layer = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet18_m = nn.Sequential(first_layer,*(list(resnet18.children())[1:-1]))
            self.backbone = resnet18_m
            self.pre_classification = nn.Sequential(nn.Dropout(),nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU())
            self.classification = nn.Linear(512,num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # TODO: Implement forward pass
        label = batch['label']
        vision_data = batch['vision']
        vision_data = self.backbone(vision_data)
        vision_data = torch.flatten(vision_data,1)
        vision_data = self.pre_classification(vision_data)
        vision_data = self.classification(vision_data)
        vision_data = self.softmax(vision_data)
        #print(vision_data)
        loss = self.loss(vision_data,label)
        _, predicted = torch.max(vision_data, 1)
        #print(predicted)
        #print(vision_data.shape)
        #print(label.shape)
        #print(predicted.shape)
        #print(loss)
        dictionary = {}
        dictionary["loss"] = loss
        dictionary["pred"] = vision_data
        return dictionary
        



class AudioClassifier(nn.Module):
    """Audio-based material classifier
    
    Instructions:
    1. Similar structure to VisionClassifier
    2. Modify first conv layer for spectrogram input (1 channel)
    3. Add frequency attention mechanism
    4. Support both ResNet and FENet backbones
    """
    def __init__(self, num_classes, backbone='resnet'):
        super().__init__()
        # TODO: Initialize network
        if backbone == "resnet":
            print("audio using resnet")
            resnet18 = models.resnet18(pretrained = True)
            first_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet18_m = nn.Sequential(first_layer,*(list(resnet18.children())[1:-1]))
            self.backbone = resnet18_m
            self.pre_classification = nn.Sequential(nn.Dropout(),nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU())
            self.classification = nn.Linear(512,num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
        self.attention = nn.MultiheadAttention(128,1)
    def forward(self, batch):
        # TODO: Implement forward pass
        label = batch['label']
        audio_data = batch['audio']
        audio_data = nn.functional.interpolate(audio_data,[256,256],mode='bilinear', align_corners=True)
        audio_data = self.backbone(audio_data)
        audio_data = torch.flatten(audio_data,1)
        audio_data = self.pre_classification(audio_data)
        #audio_data = self.attention(audio_data,audio_data,audio_data)
        audio_data = self.classification(audio_data)
        audio_data = self.softmax(audio_data)
        #print(vision_data)
        loss = self.loss(audio_data,label)
        _, predicted = torch.max(audio_data, 1)
        #print(predicted)
        #print(vision_data.shape)
        #print(label.shape)
        #print(predicted.shape)
        #print(loss)
        dictionary = {}
        dictionary["loss"] = loss
        dictionary["pred"] = audio_data
        return dictionary