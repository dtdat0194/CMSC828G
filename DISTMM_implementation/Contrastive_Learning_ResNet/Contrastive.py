'''
Coding assignment from CMSC848M-0101: Selected Topics in Information Processing; Multimodal Computer Vision
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unimodal import VisionClassifier, TouchClassifier, AudioClassifier

class ModalityEncoder(nn.Module):
    """Encoder network for each modality
    
    Instructions:
    1. Initialize backbone with pretrained weights
    2. Add projection head for contrastive learning
    3. Freeze backbone parameters
    4. Handle different input modalities
    
    Args:
        modality (str): Modality type ('vision', 'touch', or 'audio')
        output_dim (int): Dimension of output features
        num_classes (int): Number of classes for the classifier
    """
    def __init__(self, modality, output_dim, num_classes):
        super().__init__()
        # TODO: Initialize backbone and projector
        if modality == 'vision':
            self.model = VisionClassifier(num_classes=7)
            self.load_pretrained_weights('vision')

            backbone = nn.Sequential(*(list(self.model.children())[0:1]))
            classification_head = nn.Sequential(*(list(self.model.children())[1:-3]))
            self.backbone = backbone
            self.classification_head = classification_head

        else:
            self.model = AudioClassifier(num_classes=7)
            self.load_pretrained_weights('audio')

            backbone = nn.Sequential(*(list(self.model.children())[0:1]))
            classification_head = nn.Sequential(*(list(self.model.children())[1:-4]))
            self.backbone = backbone
            self.classification_head = classification_head

        
        projection = nn.Linear(512,output_dim)
        batch_norm1d = nn.BatchNorm1d(output_dim)
        ReLU = nn.ReLU()
        self.projection_layer = nn.Sequential(projection,batch_norm1d,ReLU)

        
    def load_pretrained_weights(self, modality):
        """Load pretrained weights for backbone
        
        Instructions:
        1. Load correct checkpoint for each modality
        2. Update model state dict
        3. Handle loading errors
        """
        base_path = '/cmlscratch/xyu054/MMCV/cmsc848M/experiments/unimodal'
        if modality == 'vision':
            ckpt = torch.load(f'{base_path}/visual_exp/best_model.pth', weights_only=False)
        else:
            ckpt = torch.load(f'{base_path}/audio_exp/best_model.pth', weights_only=False)
        # TODO: Load state dict and handle errors

        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Forward pass
        
        Instructions:
        1. Extract features from backbone
        2. Handle different feature dimensions
        3. Project features to contrastive space
        4. Normalize projected features
        
        Returns:
            tuple: (features, projected_features)
        """
        # TODO: Implement forward pass
        x = self.backbone(x)
        x = torch.flatten(x,1)
        x = self.classification_head(x)
        x1 = self.projection_layer(x)
        x1 = nn.functional.normalize(x1)
        return (x,x1)

class ContrastiveLearning(nn.Module):
    """Contrastive learning framework
    
    Instructions:
    1. Initialize encoders for each modality
    2. Implement contrastive loss calculation
    3. Handle both intra-modal and cross-modal pairs
    4. Support both training and inference modes
    
    Args:
        modalities (list): List of modalities to use
        feature_dim (int): Feature dimension
        temperature (float): Temperature for scaling
        num_classes (int): Number of classes for the classifier
    """
    def __init__(self, modalities, feature_dim, temperature=0.07, num_classes=7):
        super().__init__()
        # TODO: Initialize encoders and parameters
        self.vision_encoder = ModalityEncoder("vision", output_dim = feature_dim, num_classes = num_classes)
        self.touch_encoder = ModalityEncoder("touch", output_dim = feature_dim, num_classes = num_classes)
        self.audio_encoder = ModalityEncoder("audio", output_dim = feature_dim, num_classes = num_classes)
        
        self.temperature = temperature
    def forward(self, batch):
        """Forward pass
        
        Instructions:
        1. Process each modality pair
        2. Calculate intra-modal and cross-modal losses
        3. Combine all losses with proper weighting
        4. Return features for downstream tasks
        
        Returns:
            dict: Contains loss, features, projections
        """
        # TODO: Implement forward pass
        label = batch['label']
        vision_data_1 = batch['vision_1']
        vision_data_1 = self.vision_encoder(vision_data_1)
        vision_features_1 = vision_data_1[0]
        vision_projections_1 = vision_data_1[1]

        vision_data_2 = batch['vision_2']
        vision_data_2 = self.vision_encoder(vision_data_2)
        vision_features_2 = vision_data_2[0]
        vision_projections_2 = vision_data_2[1]


        
        touch_data_1 = batch['touch_1']
        touch_data_1 = nn.functional.interpolate(touch_data_1,[256,256],mode='bilinear', align_corners=True)
        touch_data_1 = self.touch_encoder(touch_data_1)
        touch_features_1 = touch_data_1[0]
        touch_projections_1 = touch_data_1[1]

        touch_data_2 = batch['touch_2']
        touch_data_2 = nn.functional.interpolate(touch_data_2,[256,256],mode='bilinear', align_corners=True)
        touch_data_2 = self.touch_encoder(touch_data_2)
        touch_features_2 = touch_data_2[0]
        touch_projections_2 = touch_data_2[1]

        audio_data_1 = batch['audio_1']
        audio_data_1 = nn.functional.interpolate(audio_data_1,[256,256],mode='bilinear', align_corners=True)
        audio_data_1 = self.audio_encoder(audio_data_1)
        audio_features_1 = audio_data_1[0]
        audio_projections_1 = audio_data_1[1]

        audio_data_2 = batch['audio_2']
        audio_data_2 = nn.functional.interpolate(audio_data_2,[256,256],mode='bilinear', align_corners=True)
        audio_data_2 = self.audio_encoder(audio_data_2)
        audio_features_2 = audio_data_2[0]
        audio_projections_2 = audio_data_2[1]

        loss = self.info_nce_loss(vision_projections_1,vision_projections_2,label)
        loss += self.info_nce_loss(vision_projections_1,touch_projections_1,label)
        loss += self.info_nce_loss(vision_projections_1,touch_projections_2,label)
        loss += self.info_nce_loss(vision_projections_1,audio_projections_1,label)
        loss += self.info_nce_loss(vision_projections_1,audio_projections_2,label)

        loss += self.info_nce_loss(vision_projections_2,touch_projections_1,label)
        loss += self.info_nce_loss(vision_projections_2,touch_projections_2,label)
        loss += self.info_nce_loss(vision_projections_2,audio_projections_1,label)
        loss += self.info_nce_loss(vision_projections_2,audio_projections_2,label)

        loss += self.info_nce_loss(touch_projections_1,touch_projections_2,label)
        loss += self.info_nce_loss(touch_projections_1,audio_projections_1,label)
        loss += self.info_nce_loss(touch_projections_1,audio_projections_2,label)

        loss += self.info_nce_loss(touch_projections_2,audio_projections_1,label)
        loss += self.info_nce_loss(touch_projections_2,audio_projections_2,label)

        loss += self.info_nce_loss(audio_projections_1,audio_projections_2,label)

        loss = loss/15

        dictionary = {}
        dictionary["loss"] = loss

        dictionary["vision_features_1"] = vision_features_1
        dictionary["vision_projections_1"] = vision_projections_1

        dictionary["vision_features_2"] = vision_features_2
        dictionary["vision_projections_2"] = vision_projections_2

        dictionary["touch_features_1"] = touch_features_1
        dictionary["touch_projections_1"] = touch_projections_1

        dictionary["touch_features_2"] = touch_features_2
        dictionary["touch_projections_2"] = touch_projections_2

        dictionary["audio_features_1"] = audio_features_1
        dictionary["audio_projections_1"] = audio_projections_1

        dictionary["audio_features_2"] = audio_features_2
        dictionary["audio_projections_2"] = audio_projections_2

        return dictionary

    def info_nce_loss(self, features_1, features_2, labels):
        """Calculate improved InfoNCE loss
        
        Instructions:
        1. Compute similarity matrix
        2. Apply temperature scaling
        3. Handle positive and negative pairs
        4. Implement hard negative mining
        
        Returns:
            torch.Tensor: Calculated contrastive loss
        """
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