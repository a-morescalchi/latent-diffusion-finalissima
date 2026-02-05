''''
COntains the train() function for training the LoRA
'''

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import os


from ldm.util import instantiate_from_config
from lora_train_dataset import LoRADataset
from lora import loraModel

# --- Configuration ---
CONFIG_PATH = "configs/latent-diffusion/celebahq-ldm-vq-4.yaml" # Check this path!
CKPT_PATH = "celeba/model.ckpt" # Check this path!
OUTPUT_DIR = "lora_checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_URL = "hf://datasets/huggan/few-shot-obama/data/train-00000-of-00001.parquet"



def load_model(config_path, ckpt_path):

    config = OmegaConf.load(config_path)
    
    if "first_stage_config" in config.model.params:
        if "ckpt_path" in config.model.params.first_stage_config.params:
            print("Patched config: Removing reference to missing first_stage_model.")
            del config.model.params.first_stage_config.params["ckpt_path"]

    model = instantiate_from_config(config.model)

    print(f"Loading weights from {ckpt_path}...")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    m, u = model.load_state_dict(sd, strict=False)
    
    if len(m) > 0:
        print(f"Missing keys: {len(m)}")
    if len(u) > 0:
        print(f"Unexpected keys: {len(u)}")
        
    model.to(DEVICE)
    model.eval()
    return model


def train(BATCH_SIZE = 4, LR = 1e-4, EPOCHS = 400, rank=16, alpha=64, qkv=[True, True, True]):
    ''''
    BATCH_SIZE, LR, EPOCHS are trainign hyperparameters
    rank, alpha, qkv are LoRA hyperparameters
    '''
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    

    model = load_model(CONFIG_PATH, CKPT_PATH)
    
    unet = model.model.diffusion_model
    
    unet = loraModel(unet, rank=rank, alpha=alpha, qkv=qkv)
    unet.to(DEVICE)
    unet.set_trainable_parameters()
    model.model.diffusion_model = unet
    

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-2)
    initial_weights = {}
    for name, param in unet.named_parameters():
        if 'lora' in name and param.requires_grad:
            initial_weights[name] = param.data.clone()

    dataset = LoRADataset(parquet_url=DATASET_URL, size=256)
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    unet.train()  
    model.eval() 
    
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x in pbar:
            x = x.to(DEVICE)
            
            with torch.no_grad():
                z = model.get_first_stage_encoding(model.encode_first_stage(x))
            
            t = torch.randint(0, model.num_timesteps, (x.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(z)
            
            x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
            
            model_output = model.apply_model(x_noisy, t, cond=None)
            
            loss = F.mse_loss(model_output, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            grad_norms = [p.grad.norm().item() for p in trainable_params if p.grad is not None]
            if len(grad_norms) > 0:
                avg_grad = sum(grad_norms) / len(grad_norms)
                pbar.set_postfix(loss=loss.item(), grad=f"{avg_grad:.4f}")
            else:
                print("WARNING: No gradients")

            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())


        torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, f"lora_epoch_{epoch}.pt"))

        if epoch % 10 == 0:
            weight_changes = []
            for name, param in unet.named_parameters():
                if 'lora' in name and param.requires_grad:
                    change = (param.data - initial_weights[name]).abs().mean().item()
                    weight_changes.append(change)
            avg_change = sum(weight_changes) / len(weight_changes) if weight_changes else 0
            print(f"Epoch {epoch}: Average LoRA weight change: {avg_change:.6f}")
            
        avg_change = sum(weight_changes) / len(weight_changes) if weight_changes else 0
        print(f"Epoch {epoch}: Average LoRA weight change: {avg_change:.6f}")


    torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, f"lora_final.pt"))
    print('Weights saved in ', OUTPUT_DIR+"/lora_final.pt")

if __name__ == "__main__":
    train()