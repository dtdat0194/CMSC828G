import torch
from dataset import get_dataloader
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os
import cv2

def visualize_samples(root_dir, num_samples=5):
    """Visualize the first 5 samples from each category"""
    # Get file lists
    video_dir = os.path.join(root_dir, 'video')
    audio_dir = os.path.join(root_dir, 'audio')
    spec_dir = os.path.join(root_dir, 'spectrograms')
    
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])[:num_samples]
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])[:num_samples]
    spec_files = sorted([f for f in os.listdir(spec_dir) if f.endswith('.png')])[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3*num_samples))
    fig.suptitle('First 5 Samples from Each Category', fontsize=16)
    
    # Process each sample
    for i in range(num_samples):
        # Load and display video frame
        video_path = os.path.join(video_dir, video_files[i])
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[i, 0].imshow(frame)
            axes[i, 0].set_title(f'Video: {video_files[i]}')
        axes[i, 0].axis('off')
        cap.release()
        
        # Load and display audio waveform
        audio_path = os.path.join(audio_dir, audio_files[i])
        audio, sr = librosa.load(audio_path, sr=None)
        time = np.linspace(0, len(audio)/sr, len(audio))
        axes[i, 1].plot(time, audio)
        axes[i, 1].set_title(f'Audio: {audio_files[i]}')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('Amplitude')
        
        # Load and display spectrogram
        spec_path = os.path.join(spec_dir, spec_files[i])
        spec = plt.imread(spec_path)
        axes[i, 2].imshow(spec)
        axes[i, 2].set_title(f'Spectrogram: {spec_files[i]}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    root_dir = 'micro-clips'
    
    # Print directory structure
    print("\nDirectory Structure:")
    for dir_name in ['video', 'audio', 'spectrograms']:
        dir_path = os.path.join(root_dir, dir_name)
        files = os.listdir(dir_path)
        print(f"\n{dir_name.upper()} directory ({len(files)} files):")
        for f in sorted(files)[:5]:  # Show first 5 files
            print(f"- {f}")
    
    # Visualize samples
    visualize_samples(root_dir, num_samples=5)

if __name__ == "__main__":
    main() 
