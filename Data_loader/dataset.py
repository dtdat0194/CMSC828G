import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
#import librosa
import torchaudio
class MicroClipsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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
        if torch.is_tensor(idx):
            idx = idx.tolist()

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
        cap = cv2.VideoCapture(video_path)
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video frame from {video_path}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for transforms
        frame = Image.fromarray(frame)
        
        # Apply transforms
        if self.transform:
            frame = self.transform(frame)
        
        cap.release()
        
        # Load corresponding audio
        audio_name = f"{file_id}_audio_{clip_index}.wav"
        audio_path = os.path.join(self.audio_dir, audio_name)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)  
        #audio = torch.from_numpy(audio).float() 
        
        # Load corresponding spectrogram
        spec_name = f"{file_id}_audio_{clip_index}_spectrogram.png"
        spec_path = os.path.join(self.spectrogram_dir, spec_name)
        
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Spectrogram not found: {spec_path}")
        
        # Load spectrogram
        spectrogram = Image.open(spec_path)
        spectrogram = transforms.ToTensor()(spectrogram)
        
        return {
            'video_frame': frame,
            'audio': audio,
            'audio_sr': sr,
            'spectrogram': spectrogram,
            'video_name': video_name
        }

def get_dataloader(root_dir, batch_size=32, shuffle=True, num_workers=4):
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
    dataset = MicroClipsDataset(root_dir=root_dir)
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
        batch_size=1,
        shuffle=True
    )
    
    # Test the dataloader
    for batch in dataloader:
        print(f"Video frame shape: {batch['video_frame'].shape}")
        print(f"Audio shape: {batch['audio'].shape}")
        print(f"Audio sample rate: {batch['audio_sr']}")
        print(f"Spectrogram shape: {batch['spectrogram'].shape}")
        print(f"Video names: {batch['video_name']}")
        break  # Just test first batch 


# Example usage
if __name__ == "__main__":
    # Create dataloader
    main()
    
