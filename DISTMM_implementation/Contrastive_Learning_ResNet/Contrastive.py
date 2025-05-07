'''
Based on Coding assignment from CMSC848M-0101: Selected Topics in Information Processing; Multimodal Computer Vision
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from Backbones import VisionClassifier, TouchClassifier, AudioClassifier

class ModalityEncoder(nn.Module):
    def __init__(self, modality, output_dim):
        super().__init__()
        # TODO: Initialize backbone and projector
        if modality == 'vision':
            self.backbone = VisionClassifier()

        else:
            self.backbone = AudioClassifier()
        projection = nn.Linear(512,output_dim)
        batch_norm1d = nn.BatchNorm1d(output_dim)
        ReLU = nn.ReLU()
        self.projection_layer = nn.Sequential(projection,batch_norm1d,ReLU)

    def forward(self, x):
        # TODO: Implement forward pass
        x = self.backbone(x)
        x = torch.flatten(x,1)
        x = self.projection_layer(x)
        return x

class ContrastiveLearning(nn.Module):
    def __init__(self, modalities, feature_dim = 128, temperature=0.07):
        super().__init__()
        # TODO: Initialize encoders and parameters
        self.vision_encoder = ModalityEncoder("vision", output_dim = feature_dim)
        self.audio_encoder = ModalityEncoder("audio", output_dim = feature_dim)
        self.temperature = temperature

    def forward(self, batch):
        # TODO: Implement forward pass
        label = batch['label']
        video = batch['video']
        projected_video = self.vision_encoder(video)
        
        audio = batch['audio']
        projected_audio = self.audio_encoder(audio)
        

        loss = self.info_nce_loss(video,audio,label)
        loss += self.info_nce_loss(audio,video,label)
        
        loss = loss/2

        return projected_audio, projected_video

    def info_nce_loss(self, features_1, features_2, labels):
        # TODO: Implement InfoNCE loss
        #Inspired from https://github.com/arashkhoeini/infonce/blob/main/infonce/infonce.py

        similarity = torch.matmul(features_1,features_2.t()) / self.temperature

        
        pos_mask_matrix = (labels.unsqueeze(1) == labels.t().unsqueeze(0)).float()
        neg_mask_matrix = 1-pos_mask_matrix

        pos_mask_add = neg_mask_matrix * (-1000)
        neg_mask_add = pos_mask_matrix * (-1000)


        loss = torch.logsumexp((similarity * pos_mask_matrix+pos_mask_add),-1) - torch.logsumexp(similarity ,-1)
        loss = -torch.mean(loss)
        #print(loss)
        return loss
