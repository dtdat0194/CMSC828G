'''
Coding assignment from CMSC848M-0101: Selected Topics in Information Processing; Multimodal Computer Vision
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchsummary import summary#check info of the model


class VisionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize backbone and classifier
        '''
        resnet152 = models.resnet152(weights='DEFAULT')
        first_layer = nn.Conv3d(3, 64, kernel_size=(2,7,7), stride=2, padding=3, bias=False)
        resnet152_m = nn.Sequential(first_layer,*(list(resnet152.children())[1:-1]))
        self.backbone = resnet152_m
        self.pre_classification = nn.Sequential(nn.Dropout(),nn.Linear(2048,512),nn.BatchNorm1d(512),nn.ReLU())
        '''
        resnet152 = models.resnet152(weights='DEFAULT')
        self.first_layer = nn.Conv3d(3, 64, kernel_size=(2,7,7), stride=2, padding=(0,3,3), bias=False)
        resnet152_m = nn.Sequential(*(list(resnet152.children())[1:-1]))
        self.backbone = resnet152_m
        self.attention1 = nn.MultiheadAttention(2048,16)
        self.linear1 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention2 = nn.MultiheadAttention(2048,16)
        self.linear2 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention3 = nn.MultiheadAttention(2048,16)
        self.linear3 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention4 = nn.MultiheadAttention(2048,16)
        self.linear4 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention5 = nn.MultiheadAttention(2048,16)
        self.linear5 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention6 = nn.MultiheadAttention(2048,16)
        self.linear6 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention7 = nn.MultiheadAttention(2048,16)
        self.linear7 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention8 = nn.MultiheadAttention(2048,16)
        self.linear8 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention9 = nn.MultiheadAttention(2048,16)
        self.linear9 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention10 = nn.MultiheadAttention(2048,16)
        self.linear10 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention11 = nn.MultiheadAttention(2048,16)
        self.linear11 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention12 = nn.MultiheadAttention(2048,16)
        self.linear12 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.attention13 = nn.MultiheadAttention(2048,16)
        self.linear13 = nn.Sequential(nn.ReLU(),nn.Linear(2048,2048),nn.BatchNorm1d(2048),nn.ReLU(),nn.Dropout())
        self.pre_classification = nn.Sequential(nn.Linear(2048,512),nn.BatchNorm1d(512),nn.ReLU())
    def forward(self, batch):
        # TODO: Implement forward pass
        '''
        label = batch['label']
        vision_data = batch['vision']
        '''
        '''
        vision_data = batch
        print(1)
        print(vision_data.shape)
        vision_data = self.backbone(vision_data)
        vision_data = torch.flatten(vision_data,1)
        vision_data = self.pre_classification(vision_data)
        '''
        vision_data = batch
        vision_data = self.first_layer(vision_data)
        vision_data = torch.permute(vision_data , (2,0,1,3,4))
        vision_data = vision_data [0]
        vision_data = self.backbone(vision_data)
        vision_data = torch.flatten(vision_data,1)


        vision_data = self.attention1(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear1(vision_data)
        vision_data = self.attention2(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear2(vision_data)
        vision_data = self.attention3(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear3(vision_data)
        vision_data = self.attention4(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear4(vision_data)
        vision_data = self.attention5(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear5(vision_data)
        vision_data = self.attention6(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear6(vision_data)
        vision_data = self.attention7(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear7(vision_data)
        vision_data = self.attention8(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear8(vision_data)
        vision_data = self.attention9(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear9(vision_data)
        vision_data = self.attention10(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear10(vision_data)
        vision_data = self.attention11(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear11(vision_data)
        vision_data = self.attention12(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear12(vision_data)
        vision_data = self.attention13(vision_data,vision_data,vision_data)[0]
        vision_data = self.linear13(vision_data)

        
        vision_data = self.pre_classification(vision_data)
 

        return vision_data


class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize network
        
        print("audio using resnet")
        resnet18 = models.resnet18(weights="DEFAULT")
        first_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet18_m = nn.Sequential(first_layer,*(list(resnet18.children())[1:-1]))
        self.backbone = resnet18_m
        self.pre_classification = nn.Sequential(nn.Dropout(),nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU())
 
    def forward(self, batch):
        # TODO: Implement forward pass
        #label = batch['label']
        #audio_data = batch['audio']

        audio_data = batch
        audio_data = nn.functional.interpolate(audio_data,[256,256],mode='bilinear', align_corners=True)

        #print("audio_data shape", audio_data.shape)
        audio_data = self.backbone(audio_data)
        audio_data = torch.flatten(audio_data,1)
        audio_data = self.pre_classification(audio_data)
        return audio_data

'''
model = VisionClassifier()
model.to("cuda")
summary(model, (3,2, 256, 256))
'''