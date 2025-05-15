import os, sys, tempfile, argparse

from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from Backbones import VisionClassifier, AudioClassifier
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


#Creating Distributed-setup helpers
def setup_dist():

    #using pytorchs example of how to set this up.
    #if we have a windows platform well us the gloo backed + Filestor
    if sys.platform == "win32":
        if"INIT_METHOD" in os.environ:
            url = urlparse(os.environ["INIT_METHOD"])
            if url.scheme.lower() != "file":
                raise ValueError("Windows supports only FileStore on Gloo")
            init_method = os.environ["INIT_METHOD"]
        else:
            init_method = f"file:///{os.path.join(tempfile.gettempdir(), 'ddp_example')}"

        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),

        )
    else:
        #This says it's unreachable and I'm not exactly sure why, it looks fine to me.
        #Going to use NCCL with all parameters supplied by torchrun.
        env = {k: os.getenv(k) for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")}
        print(f"[{os.getpid()}] Initialising process group with: {env}")
        dist.init_process_group(backend="nccl")

    #just making sure that torch cuda gpus are avaliable, we might want to remove this if it goofs
    #our run for some reason.
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        print(
            f"[{os.getpid()}] backend={dist.get_backend()}  "
            f"rank={dist.get_rank()}/{dist.get_world_size()}  "
            f"device={torch.cuda.current_device()}"
        )

def clean_dist():
    #destroying our defualt process group
    dist.destroy_process_group()


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
        self.vision_encoder = ModalityEncoder("vision", output_dim=feature_dim)

        self.audio_encoder = ModalityEncoder("audio", output_dim=feature_dim)

        self.audio_encoder.to("cuda:3")
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
        return video_data, spectrogram_data, loss
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
    
    def parse_args():
        parser = argparse.ArgumentParser()

        #setting LOCL_Rank and WORLD_SIZE in the env, and grabbing local_rank for pinning to CUDA
        parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
        parser.add_argument("--local_world_size", type=int, default=1)
        return parser.parse_args()
        