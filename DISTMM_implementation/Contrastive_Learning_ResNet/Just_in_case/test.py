import os 
import torch
import torch.nn as nn
import torch.nn.functional as F

from Backbones import VisionClassifier, AudioClassifier
import utils
from dataset import get_dataloader
def setup_optimization(self):
        """Setup optimizer and learning rate scheduler"""
        if self.args.finetune:
            # Different learning rates for pretrained layers and new layers
            backbone_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.args.lr * 0.1},
                {'params': head_params, 'lr': self.args.lr}
            ], weight_decay=self.args.weight_decay)
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6
        )
def train_epoch(self, epoch):

        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        if self.args.task == 'contrastive':
            # contrastive learning training loop
            for batch in self.dataloaders['train']:
                # move data to device
                for modality in self.args.modality_list:
                    batch[f'{modality}_1'] = batch[f'{modality}_1'].to(self.device)
                    batch[f'{modality}_2'] = batch[f'{modality}_2'].to(self.device)
                batch['label'] = batch['label'].to(self.device)
                
                # forward propagation
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs['loss']
                
                # backward propagation
                loss.backward()
                self.optimizer.step()
                
                # only record loss
                running_loss += loss.item()
        else:
            # original classification training loop
            for batch in self.dataloaders['train']:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs['loss']
                preds = outputs['pred']
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                all_preds.append(preds.detach())
                all_labels.append(batch['label'])
        
        # calculate average loss
        epoch_loss = running_loss / len(self.dataloaders['train'])
        
        if self.args.task == 'contrastive':
            # contrastive learning only record loss
            self.train_losses.append(epoch_loss)
            accuracy = 0
            mAP = 0
        else:
            # classification task calculate all metrics
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            label_one_hot = torch.zeros(all_labels.size(0), self.cfg.num_classes)
            label_one_hot.scatter_(1, all_labels.cpu().unsqueeze(1), 1)
            
            accuracy = (all_preds.argmax(dim=1) == all_labels).float().mean() * 100
            mAP = average_precision_score(
                label_one_hot.numpy(),
                all_preds.cpu().numpy()
            ) * 100
            
            self.train_losses.append(epoch_loss)
            self.train_accs.append(accuracy.item())
            self.train_maps.append(mAP)
        
        return epoch_loss


video_dataset_path = "/cmlscratch/xyu054/DistMM/Contrastive_Learning_ResNet/dataset/clips"
audio_dataset_path = "/cmlscratch/xyu054/DistMM/Contrastive_Learning_ResNet/dataset/audio"
video_name_list = os.listdir(video_dataset_path)
audio_name_list = os.listdir(audio_dataset_path)

