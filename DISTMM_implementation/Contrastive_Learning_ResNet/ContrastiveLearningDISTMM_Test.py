#Trying to figure out DDP and get it running from pytorches notes.

#Launch with the command 
# torchrun --nproc_per_node=4 ContrastiveLearningDISTMM_Test.py --epochs 50

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

class ContrastiveLearning(nn.Module):
    def __init__(self, feature_dim = 128, temperature=0.07):
        super().__init__()
        # TODO: Initialize encoders and parameters

        self.temperature = temperature
        self.rank = dist.get_rank()
        self.audio_rank = audio_rank

        self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        self.vision_encoder = ModalityEncoder("vision", output_dim=feature_dim)

        self.audio_encoder = ModalityEncoder("audio", output_dim=feature_dim)

    def forward(self, video_data: torch.Tensor, spectrogram_data: torch.Tensor):
        '''
        video_features = self.vision_encoder(video_data)
        audio_features = self.audio_encoder(spectrogram_data)
        '''
        #device = next(self.parameters()).device

        #trying to communicate across all to get our loss function.
        video_data = video_data.to(self.device, non_blocking=True)
        spectogram = spectrogram_data.to(self.device, non_blocking=True)

        #making sure that I'm doing a fowardpass on every rank(GPU) for my vision
        video_features = self.vision_encoder(video_data)
        batch_size = spectogram.size(0)

        video_features = video_features.view(batch_size, 3, -1).mean(1) #average over our 3 GPUs

        #Forward pass for my audio will only happen on GPU 3.
        if self.rank == 3:
            audio_features = self.audio_encoder(spectogram)
        else:
            audio_features = torch.zeros(B, video_features.size(1), device=self.device, dtype=video_features.dtype)

        #broadcast audio features across all ranks from rank 3.
        dist.broadcast(audio_features, src=3)

        #gathering all of my video features, so every rank can see the global batch.
        video_gather_list = [torch.empty_like(video_features) for _ in range (dist.get_world_size())]
        dist.all_gather(video_gather_list, video_features, async_op=False)
        #doing it on each rank
        video_all = torch.cat(video_gather_list, dim=0)
        audio_all = audio_features.repeat(dist.get_world_size(), 1)

        #calling our info_nce_loss
        loss = self.info_nce_loss(video_all,audio_all)
        return loss

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
    
    #doing a training step, this is just rough, will need to make modifications to this.

def train_one_step(model, optim, batch_size=8):
    # (local shard) dummy data, wanna talk about this a little more to see if we can modify it, if
    #not we can remove it. But I'm unsure how to do this without having it live outside
    #of our current setup.
    vid  = torch.randn(batch_size * 3, 3, 256, 256, device=model.device)
    spec = torch.randn(batch_size, 1, 128, 128, device=model.device)

    loss = model(vid, spec)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()

    #my main function that will call everything and do my wrapping of my DDP

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        #setting LOCL_Rank and WORLD_SIZE in the env, and grabbing local_rank for pinning to CUDA
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    parser.add_argument("--local_world_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args() 


    setup_dist()

    local_rank = args.local_rank
    world_size = dist.get_world_size()
    if args.audio_rank < 0:
        audio_rank = world_size - 1
    else:
        audio_rank = args.audio_rank

    #setup_dist()
    #device = torch.cuda.current_device()

    model = ContrastiveLearning(feature_dim=128, temperature=0.07, audio_rank=audio_rank)

    # 1) select and pin CUDA device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # 2) move model (and all submodules) onto that GPU
    model.to(device)

    #Turning unused parameters on since audio isn't used on GPUs 0,1,2
    contrastivelearningDDP_model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                    output_device=int(os.environ["LOCAL_RANK"]),
                    find_unused_parameters=True)
    
    optimizer = torch.optim.Adam(contrastivelearningDDP_model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        #optimizer.zero_grad()
        loss_val = train_one_step(contrastivelearningDDP_model, optimizer)
        #optimizer.step()
        if dist.get_rank() == 0:
            print(f"Epoch {epoch:03d} | loss = {loss_val:.4f}")

    clean_dist()

        