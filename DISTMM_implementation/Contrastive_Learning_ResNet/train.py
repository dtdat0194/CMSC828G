import os
import os.path as osp
import argparse
import yaml
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time

class MaterialTrainer:
    """Training and evaluation framework for material classification
    
    This class handles:
    1. Single modality classification (Task 1)
    2. Multi-modal fusion (Task 2)
    3. Contrastive learning (Task 3)
    """
    def __init__(self,audio_gpu_device,video_gpu_devices,device):
        self.device = device
        self.audio_gpu_device = audio_gpu_device
        self.video_gpu_devices = video_gpu_devices
        self.setup_environment()
        self.build_dataloaders()
        self.build_model()
        self.setup_optimization()
        self.train_losses = []
        self.train_accs = []
        self.train_maps = []
        self.val_losses = []
        self.val_accs = []
        self.val_maps = []
        self.epoch_data_loading_time = 0
        self.epoch_forward_time = 0
        self.epoch_backward_time = 0
        self.epoch_step_time = 0
    def setup_environment(self):
        """Setup random seeds and computing device"""
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        

    def build_dataloaders(self):
        """Initialize train/val/test dataloaders"""
        from dataset import get_dataloader
        self.dataloader = get_dataloader(
        root_dir='/cmlscratch/xyu054/DistMM/Contrastive_Learning_ResNet/dataset/micro-clips',
        audio_gpu_device = self.audio_gpu_device,
        video_gpu_devices = self.video_gpu_devices,
        batch_size=32,
        shuffle=False,
        num_workers = 16
    )
        

    def build_model(self):
        """Initialize model based on task type"""
        from Contrastive import ContrastiveLearning
        self.model = ContrastiveLearning().to(self.device)


    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=1e-2
            )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=20,#self.args.epochs
            eta_min=1e-6
        )


    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0

        self.last_end_time = self.epoch_start
        for batch in tqdm(self.dataloader):
            batch_start_time = time.time() - self.last_end_time
            print()
            print(f"Data loading takes {batch_start_time} seconds")
            self.epoch_data_loading_time += batch_start_time
            video_data = batch['video_data']
            spectrogram_data = batch['spectrogram']
            file_id = batch['file_id']

            B, S = video_data.shape[:2]
            video_data = video_data.reshape(
                    B * S, *video_data.shape[2:]
                )
            B, S = spectrogram_data.shape[:2]
            spectrogram_data = spectrogram_data.reshape(
                    B * S, *spectrogram_data.shape[2:]
                )

            video_data = video_data.to(self.video_gpu_devices)
            #print(video_data.shape)
            spectrogram_data = spectrogram_data.to(self.audio_gpu_device)
            # forward propagation
            self.optimizer.zero_grad()
            forward_start = time.time()
            _,_,loss = self.model(video_data,spectrogram_data)
            #print("loss:",loss)
            forward_end = time.time()
            forward_time = forward_end -forward_start
            print(f"Forward uses {forward_time} seconds")
            self.epoch_forward_time += forward_time
            # backward propagation
            loss.backward()
            backword_end = time.time()
            backword_time = backword_end - forward_end
            self.epoch_backward_time += backword_time
            print(f"Backward uses {backword_time} seconds")
            self.optimizer.step()
            step_end = time.time()
            step_time = step_end - backword_end
            self.epoch_step_time += step_time
            print(f"Step uses {step_time} seconds")
            # only record loss
            running_loss += loss.item()
            print("Batch ends.")
            print("-----------------------------")
            print()
            self.last_end_time = time.time()
        
        # calculate average loss
        epoch_loss = running_loss / len(self.dataloader)
        self.train_losses.append(epoch_loss)

        return epoch_loss


    def train(self):
        """Main training loop"""
        self.train_start = time.time()
        for epoch in range(1):
            # Training
            self.epoch_start = time.time() 
            print(f"Start of epoch {epoch}:",self.epoch_start - self.train_start)
            train_loss = self.train_epoch(epoch)

            # Update learning rate
            self.scheduler.step()

            print(f"Epoch {epoch}")
            print(f"Train Loss: {train_loss:.4f}")
            print()
            
            print(f"Epoch data loading time: {self.epoch_data_loading_time} seconds")
            print(f"Epoch forward time: {self.epoch_forward_time} seconds")
            print(f"Epoch backward time: {self.epoch_backward_time} seconds")
            print(f"Epoch step time: {self.epoch_step_time} seconds")
        self.end = time.time()
        self.epoch_training_time = self.end - self.train_start 
        print(f"Epoch total training time: {self.epoch_training_time} seconds")

def main():

    # Initialize trainer
    trainer = MaterialTrainer("cuda:0","cuda:0","cuda:0")
    
    # Run training or evaluation

    trainer.train()


if __name__ == '__main__':
    main()
