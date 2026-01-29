import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from ldm.models.diffusion.ddim import DDIMSampler
import numpy as np

# Imports from the latent-diffusion repo
from ldm.util import instantiate_from_config

# Import our dataset
from lora_train_dataset import LoRADataset
from lora import loraModel

# --- Configuration ---
CONFIG_PATH = "configs/latent-diffusion/celebahq-ldm-vq-4.yaml" # Check this path!
CKPT_PATH = "celeba/model.ckpt" 
OUTPUT_DIR = "lora_checkpoints"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_URL = "hf://datasets/huggan/few-shot-obama/data/train-00000-of-00001.parquet"

def load_model(config_path, ckpt_path):
    # 1. Load the YAML configuration
    config = OmegaConf.load(config_path)
    
    # --- FIX START ---
    # The config file points to a separate VQ-VAE file (models/first_stage_models/...)
    # which you don't have. We delete this key so the code doesn't try to load it yet.
    # The VAE weights will be loaded from your main 'ckpt_path' in step 3.
    if "first_stage_config" in config.model.params:
        if "ckpt_path" in config.model.params.first_stage_config.params:
            print("Patched config: Removing reference to missing first_stage_model.")
            del config.model.params.first_stage_config.params["ckpt_path"]
    # --- FIX END ---

    # 2. Instantiate the model (now with initialized, random VAE weights)
    model = instantiate_from_config(config.model)
    
    # 3. Load the actual pre-trained weights (which contain both UNet and VAE)
    print(f"Loading weights from {ckpt_path}...")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
        
    # strict=False allows loading even if there are small mismatches
    # (The main checkpoint keys will overwrite the random VAE weights)
    m, u = model.load_state_dict(sd, strict=False)
    
    if len(m) > 0:
        print(f"Missing keys: {len(m)}")
    if len(u) > 0:
        print(f"Unexpected keys: {len(u)}")
        
    model.to(DEVICE)
    return model


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model & Inject LoRA
    print("Loading model...")
    model = load_model(CONFIG_PATH, CKPT_PATH)
    
    unet = model.model.diffusion_model
    # Use alpha=rank for a scaling of 1.0 (Stronger learning signal)
    unet = loraModel(unet, rank=4, alpha=4) 
    unet.set_trainable_parameters()
    model.model.diffusion_model = unet
    model.to(DEVICE)
    
    # 2. Dataset & Optimizer
    print("Loading dataset...")
    dataset = LoRADataset(DATASET_URL)
    # We only need 1 batch for this test
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Filter params
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3) # Higher LR for overfitting test

    # --- 🧪 SCIENTIFIC OVERFIT SETUP ---
    print("\n--- 🧪 SETTING UP CONTROLLED EXPERIMENT ---")
    
    # A. Get One Batch
    x = next(iter(dataloader)).to(DEVICE)
    
    # B. Force Normalization (Critical Fix)
    if x.max() > 2.0:
        print("🔧 Auto-Fixing Data Range: Scaling [0, 255] -> [-1, 1]")
        x = (x / 127.5) - 1.0
    
    # C. Pre-Calculate Targets (Freeze the chaos)
    with torch.no_grad():
        # Encode once. (Fixes VAE sampling noise)
        z = model.get_first_stage_encoding(model.encode_first_stage(x))
        
        # Pick ONE timestep and ONE noise pattern.
        # This gives the model a single, static target to memorize.
        t = torch.tensor([400] * x.shape[0]).long().to(DEVICE) # Fixed step 400
        noise = torch.randn_like(z) # Fixed noise
        
        # Create the noisy image ONCE
        x_noisy = model.q_sample(x_start=z, t=t, noise=noise)

    print("--- STARTING OVERFIT LOOP (Target: Loss -> 0.000) ---")
    
    loss_history = []
    
    # Run 200 steps on the EXACT SAME inputs
    for epoch in range(200):
        # 1. Predict
        # We pass the FIXED x_noisy and FIXED t
        model_output = model.apply_model(x_noisy, t, cond=None)
        
        # 2. Loss
        # We compare against the FIXED noise
        loss = F.mse_loss(model_output, noise)
        
        # 3. Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

        if epoch % 5 == 0:
            print(f"\n--- Generating Progress Images (Epoch {epoch}) ---")
            
            # Switch to eval mode (disables dropout/batchnorm for sampling)
            model.eval()
            sampler = DDIMSampler(model)
            
            with torch.no_grad():
                # 1. VISUALIZE INPUT (Ground Truth Check)
                # We take the latent 'z' from the last training batch and decode it.
                # If this looks gray/static/black, your dataset loading is BROKEN.
                # If this looks like Obama, your data is correct.
                x_rec = model.decode_first_stage(z[:1]) # Decode first image of batch
                x_rec = torch.clamp((x_rec + 1.0) / 2.0, min=0.0, max=1.0) # Scale to [0, 1]
                x_rec = x_rec.cpu().numpy().transpose(0, 2, 3, 1)[0] # (H, W, C)

                # 2. VISUALIZE PROGRESS (Generation Check)
                # We generate a completely new image from random noise
                shape = z.shape[1:] # e.g. [3, 64, 64]
                samples, _ = sampler.sample(S=20, batch_size=1, shape=shape, verbose=False)
                x_gen = model.decode_first_stage(samples)
                x_gen = torch.clamp((x_gen + 1.0) / 2.0, min=0.0, max=1.0)
                x_gen = x_gen.cpu().numpy().transpose(0, 2, 3, 1)[0]
                
                # 3. DISPLAY
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                
                ax[0].imshow(x_rec)
                ax[0].set_title(f"What the Model Sees\n(Training Input)")
                ax[0].axis('off')
                
                ax[1].imshow(x_gen)
                ax[1].set_title(f"What the Model Draws\n(Epoch {epoch})")
                ax[1].axis('off')
                
                plt.show()
                plt.close()



    # Plot
    plt.plot(loss_history)
    plt.title("Overfitting Curve (Should drop to near 0)")
    plt.show()

    if loss_history[-1] < 0.05:
        print("\n✅ SUCCESS: Model overfitted! LoRA is working.")
        print("   Next Step: Switch back to full training loop (with random t/noise).")
    else:
        print("\n❌ FAILURE: Loss stuck high.")
        print("   This confirms the issue is in 'lora.py' (Architecture), not the training loop.")
    





def train(BATCH_SIZE = 4, LR = 1e-4, EPOCHS = 100, rank=4, alpha=4):
    '''
    BATCH_SIZE, LR, EPOCHS are for training loop
    rank, alpha are LoRA hyperparameters
    ''' 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load the Freeze the Base Model
    model = load_model(CONFIG_PATH, CKPT_PATH)
    
    # In CompVis/latent-diffusion, the UNet is typically at model.model.diffusion_model
    unet = model.model.diffusion_model
    
    # Freeze everything first
    #for param in model.parameters():
    #    param.requires_grad = False

    # 2. Inject Your Custom LoRA
    # ======================================================
    # ??? INSERT YOUR LORA INJECTION HERE ???
    # Example: 
    # from my_lora_implementation import inject_lora
    # inject_lora(unet, r=4) 

    unet = loraModel(unet, rank=rank, alpha=alpha, qkv=[True, True, True])
    unet.to(DEVICE)
    unet.set_trainable_parameters()

    model.model.diffusion_model = unet
    print("Checking if LoRA is in the computation graph...")
    print(f"model.model.diffusion_model type: {type(model.model.diffusion_model)}")
    print(f"Is it loraModel? {isinstance(model.model.diffusion_model, loraModel)}")
    

    # ensure only LoRA parameters have requires_grad=True
    # ======================================================
    
    # Verify we have trainable parameters
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    if len(trainable_params) == 0:
        print("WARNING: No trainable parameters found. Did you apply the LoRA?")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-2)
    initial_weights = {}
    for name, param in unet.named_parameters():
        if 'lora' in name and param.requires_grad:
            initial_weights[name] = param.data.clone()

    # 3. Dataset
    dataset = LoRADataset(parquet_url=DATASET_URL, size=256)
    
    # Check if dataset loaded correctly
    print(f"Dataset loaded: {len(dataset)} images found.")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 4. Training Loop
    model.eval()
    for mod in unet.modules():
        if 'lora' in mod.__class__.__name__.lower():
            mod.train()  # Instead of model.train()

# 1. Grab ONE batch
    fixed_batch = next(iter(loader)).to(DEVICE)
    
    # 2. Force Data Normalization (CRITICAL SAFETY)
    if fixed_batch.max() > 2.0:
        print("🔧 Fixing Data Range: [0, 255] -> [-1, 1]")
        fixed_batch = (fixed_batch / 127.5) - 1.0

    print("--- 🧪 SETUP DETERMINISTIC TARGETS ---")
    with torch.no_grad():
        # Encode the fixed image ONCE
        z = model.get_first_stage_encoding(model.encode_first_stage(fixed_batch))
        
        # 3. FREEZE TIME AND NOISE (Move this OUTSIDE the loop)
        # We pick a specific timestep (e.g., 400) and specific noise
        fixed_t = torch.full((z.shape[0],), 400, device=DEVICE, dtype=torch.long)
        fixed_noise = torch.randn_like(z)
        
        # Create the noisy latent ONCE. This is now our permanent target.
        fixed_x_noisy = model.q_sample(x_start=z, t=fixed_t, noise=fixed_noise)

    print(f"Target created. Initial Loss Check...")
    
    # 4. Training Loop
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-3) # High LR to force quick learning
    loss_history = []
    
    print("Starting Deterministic Training...")
    for epoch in range(500): # Run 500 steps
        
        # A. Predict
        # We use the SAME input (fixed_x_noisy) and SAME time (fixed_t) every single step
        model_output = model.apply_model(fixed_x_noisy, fixed_t, cond=None)
        
        # B. Loss
        # We compare against the SAME noise (fixed_noise)
        loss = F.mse_loss(model_output, fixed_noise)
        
        # C. Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

        if epoch % 25 == 0:
            print(f"\n--- Generating Progress Images (Epoch {epoch}) ---")
            
            # Switch to eval mode (disables dropout/batchnorm for sampling)
            model.eval()
            sampler = DDIMSampler(model)
            
            with torch.no_grad():
                # 1. VISUALIZE INPUT (Ground Truth Check)
                # We take the latent 'z' from the last training batch and decode it.
                # If this looks gray/static/black, your dataset loading is BROKEN.
                # If this looks like Obama, your data is correct.
                x_rec = model.decode_first_stage(z[:1]) # Decode first image of batch
                x_rec = torch.clamp((x_rec + 1.0) / 2.0, min=0.0, max=1.0) # Scale to [0, 1]
                x_rec = x_rec.cpu().numpy().transpose(0, 2, 3, 1)[0] # (H, W, C)

                # 2. VISUALIZE PROGRESS (Generation Check)
                # We generate a completely new image from random noise
                shape = z.shape[1:] # e.g. [3, 64, 64]
                samples, _ = sampler.sample(S=20, batch_size=1, shape=shape, verbose=False)
                x_gen = model.decode_first_stage(samples)
                x_gen = torch.clamp((x_gen + 1.0) / 2.0, min=0.0, max=1.0)
                x_gen = x_gen.cpu().numpy().transpose(0, 2, 3, 1)[0]
                
                # 3. DISPLAY
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                
                ax[0].imshow(x_rec)
                ax[0].set_title(f"What the Model Sees\n(Training Input)")
                ax[0].axis('off')
                
                ax[1].imshow(x_gen)
                ax[1].set_title(f"What the Model Draws\n(Epoch {epoch})")
                ax[1].axis('off')
                
                plt.show()
                plt.close()

                if epoch % 100 == 0:
                    # Save LoRA weights only
                    # You'll need to write logic to save ONLY your lora layers, not the whole model
                    torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, f"lora_overfit_epoch_{epoch}.pt"))
                    print('Weights saved in ', OUTPUT_DIR+f"/lora_epoch_{epoch}.pt")
            
            # Switch back to train mode!
            model.train()


            # 5. Result
    print(f"Final Loss: {loss_history[-1]:.6f}")
    if loss_history[-1] < 0.01:
        print("✅ SUCCESS: The model learned! (Loss dropped to near zero)")
        print("   This confirms your LoRA code is correct.")
        print("   You can now go back to full training (random t/noise), knowing the code works.")
    else:
        print("❌ FAILURE: Loss is stuck.")
        print("   This confirms the problem is in 'lora.py' (likely the normalization bug).")
        # Save LoRA weights only
        # You'll need to write logic to save ONLY your lora layers, not the whole model
        torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, f"lora_epoch_{epoch}.pt"))

    torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, f"lora_final.pt"))
    print('Weights saved in ', OUTPUT_DIR+"/lora_final.pt")

    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.show()

if __name__ == "__main__":
    train()