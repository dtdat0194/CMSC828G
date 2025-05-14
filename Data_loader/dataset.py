import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
#import librosa
import torchaudio
from utils import load_and_transform_video_data, load_and_transform_audio_data
class MicroClipsDataset(Dataset):
    def __init__(self, root_dir, audio_gpu_device,video_gpu_devices, transform=None):
        """
        Args:
            root_dir (string): Directory with all the micro-clips.
                Structure should be:
                root_dir/
                    ├── video/
                    ├── audio/
                    └── spectrograms/
            transform (callable, optional): Optional transform to be applied on video frames
        """
        self.audio_gpu_device = audio_gpu_device
        self.video_gpu_devices = video_gpu_devices

        self.root_dir = root_dir
        self.transform = transform
        
        # Get all directories
        self.video_dir = os.path.join(root_dir, 'video')
        self.audio_dir = os.path.join(root_dir, 'audio')
        self.spectrogram_dir = os.path.join(root_dir, 'spectrograms')
        
        # Get list of all video files
        self.video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.mp4')]
        
        # Sort files to ensure matching pairs
        self.video_files.sort()
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),  # Resize to standard size
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        #print(idx)
        # Get video file name
        video_name = self.video_files[idx]
        #print("video_name:",video_name)

        base_name = os.path.splitext(video_name)[0]
        #print("base_name:",base_name)
        file_id = base_name[:-8]
        clip_index = base_name[-1]
        #print("file_id:",file_id)
        #print("clip_index:",clip_index)
        #print()
        
        # Load video
        video_path = os.path.join(self.video_dir, video_name)
        print()
        print("self.video_dir",self.video_dir)
        print("video_name",video_name)
        print("video_path",video_path)
        print()
        video_data = load_and_transform_video_data([video_path],device = self.video_gpu_devices,clips_per_video = 1)
        #video_data = video_data.permute(1,0,2,3,4,5)
        video_data = video_data[0]
        '''
        B, S = video_data.shape[:2]
        video_data = video_data.reshape(
                    B * S, *video_data.shape[2:]
                )
                '''
        print()
        print("video_data loaded")
        print()
        # Load corresponding audio
        audio_name = f"{file_id}_audio_{clip_index}.wav"
        audio_path = os.path.join(self.audio_dir, audio_name)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        spectrogram_data = load_and_transform_audio_data([audio_path],device = self.audio_gpu_device,clips_per_video = 1)
        #spectrogram_data = spectrogram_data.permute(1,0,2,3,4)
        spectrogram_data = spectrogram_data[0]
        '''
        B, S = spectrogram_data.shape[:2]
        spectrogram_data = spectrogram_data.reshape(
                    B * S, *spectrogram_data.shape[2:]
                )
                '''
        return {
            'video_data': video_data,
            'spectrogram': spectrogram_data,
            'file_id': file_id
        }

def get_dataloader(root_dir, audio_gpu_device,video_gpu_devices,batch_size=32, shuffle=True, num_workers=1):
    """
    Create a DataLoader for the MicroClipsDataset
    
    Args:
        root_dir (string): Directory with all the micro-clips
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of workers for data loading
    
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    dataset = MicroClipsDataset(root_dir=root_dir,audio_gpu_device = audio_gpu_device,video_gpu_devices = video_gpu_devices)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader

def main():
    dataloader = get_dataloader(
        root_dir='/cmlscratch/xyu054/DistMM/Contrastive_Learning_ResNet/dataset/micro-clips',
        audio_gpu_device = "cuda",
        video_gpu_devices = "cuda",
        batch_size=2,
        shuffle=False,
        num_workers = 0
    )
    
    # Test the dataloader
    for batch in dataloader:
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

        print(f"Video frame shape: {video_data.shape}")
        print(f"Spectrogram shape: {spectrogram_data.shape}")
        print(f"Video names: {batch['file_id']}")
        break  # Just test first batch 


# Example usage
if __name__ == "__main__":
    # Create dataloader
    main()
    
    
