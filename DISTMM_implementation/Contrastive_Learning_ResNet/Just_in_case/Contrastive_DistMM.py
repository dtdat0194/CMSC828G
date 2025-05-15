'''
Based on Coding assignment from CMSC848M-0101: Selected Topics in Information Processing; Multimodal Computer Vision
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from Backbones import VisionClassifier, AudioClassifier

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
        '''
from torchsummary import summary
model = ModalityEncoder('vision',128)
model.to("cuda")
summary(model, (3,2, 256, 256))
'''
class ContrastiveLearning(nn.Module):
    def __init__(self, feature_dim = 128, temperature=0.07):
        super().__init__()
        # TODO: Initialize encoders and parameters
        self.vision_encoder = torch.nn.DataParallel(ModalityEncoder("vision", output_dim = feature_dim),device_ids=[0,1,2,3],output_device=3)
        self.audio_encoder = ModalityEncoder("audio", output_dim = feature_dim).to("cuda:2")
        self.temperature = temperature
    def forward(self, video_data, spectrogram_data):
        # TODO: Implement forward pass
        video_data = self.vision_encoder(video_data)
        spectrogram_data = self.audio_encoder(spectrogram_data)

        length = spectrogram_data.shape[0]
        video_data = video_data.reshape(length, 3, 128)  # group every 3 rows together   
        video_data = video_data.mean(dim=1)
        print("video_data.get_device() ",video_data.get_device() )
        print("spectrogram_data.get_device() ",spectrogram_data.get_device() )

        loss = self.info_nce_loss(video_data.to("cuda:3"),spectrogram_data.to("cuda:3"))
        return video_data, spectrogram_data,loss
    '''
    def info_nce_loss(self, z1, z2, temperature):

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = z1 @ z2.t() / temperature
        labels = torch.arange(z1.size(0), device=logits.device)

        mean_loss = F.cross_entropy(logits, labels)
        return mean_loss
    '''
    def info_nce_loss(self, features_1, features_2):
        # TODO: Implement InfoNCE loss
        #Inspired from https://github.com/arashkhoeini/infonce/blob/main/infonce/infonce.py

        similarity = torch.matmul(features_1,features_2.t()) / self.temperature

        labels = torch.arange(features_1.size(0))
        pos_mask_matrix = (labels.unsqueeze(1) == labels.t().unsqueeze(0)).float().to("cuda:3")
        neg_mask_matrix = 1-pos_mask_matrix

        pos_mask_add = neg_mask_matrix * (-1000)
        neg_mask_add = pos_mask_matrix * (-1000)


        loss = torch.logsumexp((similarity * pos_mask_matrix+pos_mask_add),-1) - torch.logsumexp(similarity ,-1)
        loss = -torch.mean(loss)
        #print(loss)
        return loss
        