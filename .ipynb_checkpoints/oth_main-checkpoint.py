import argparse
import os
import sys
import glob
import numpy as np
import time
import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from PIL import Image
import datetime
from tqdm import tqdm

sys.path.append(os.getcwd() + "/ldm")
from ldm.util import instantiate_from_config
from ldm.data import PIL_data


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, default="", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, default="", help="resume from checkpoint")
    parser.add_argument("-b", "--base", nargs="*", default=list(), help="paths to base configs")
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=True, nargs="?", help="train")
    parser.add_argument("--no-test", type=str2bool, const=True, default=True, nargs="?", help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False, help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--stage", type=str, default="0", help="training stage")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--save_freq", type=int, default=5, help="save checkpoint every N epochs")
    parser.add_argument("--log_freq", type=int, default=100, help="log every N steps")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="accumulate gradients")
    
    return parser


def seed_everything(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def save_checkpoint(model, optimizer, epoch, step, logdir, filename="checkpoint.ckpt"):
    """Save model checkpoint"""
    ckpt_path = os.path.join(logdir, "checkpoints", filename)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    return ckpt_path


def load_checkpoint(model, optimizer, ckpt_path):
    """Load model checkpoint"""
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return 0, 0
    
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove DDIM buffers if present
    ddim_keys = [k for k in state_dict.keys() if k.startswith('ddim_')]
    for key in ddim_keys:
        state_dict.pop(key, None)
    
    model.load_state_dict(state_dict, strict=False)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print("Could not load optimizer state")
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    
    print(f"Resumed from epoch {epoch}, step {global_step}")
    return epoch, global_step


def save_images(images, save_dir, global_step, epoch, batch_idx, prefix="train"):
    """Save images to disk"""
    os.makedirs(os.path.join(save_dir, "images", prefix), exist_ok=True)
    
    for k, v in images.items():
        if isinstance(v, torch.Tensor):
            grid = torchvision.utils.make_grid(v, nrow=4)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1
            grid = grid.permute(1, 2, 0).cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            
            filename = f"{k}_gs-{global_step:06d}_e-{epoch:06d}_b-{batch_idx:06d}.png"
            path = os.path.join(save_dir, "images", prefix, filename)
            Image.fromarray(grid).save(path)


class Trainer:
    def __init__(self, model, optimizer, device, logdir, log_freq=100, save_freq=5, accumulate_grad_batches=1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logdir = logdir
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.accumulate_grad_batches = accumulate_grad_batches
        self.global_step = 0
        self.current_epoch = 0
        
        # Create log directories
        os.makedirs(os.path.join(logdir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(logdir, "images"), exist_ok=True)
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            else:
                batch = batch.to(self.device)
            """
            # === FIX: Preprocess the batch ===
            if isinstance(batch, dict) and 'image' in batch and 'mask' in batch:
                # Convert from [B, H, W, C] to [B, C, H, W]
                print(batch['image'].shape,batch['masked_image'].shape,batch['mask'].shape)
                image = batch['image'].permute(0, 3, 1, 2)  # [B, 3, H, W]
                masked_image = batch['masked_image'].permute(0, 3, 1, 2)  # [B, 3, H, W]
                mask = batch['mask'].permute(0, 3, 1, 2)  # [B, 1, H, W]
                
                # Concatenate to create 7-channel input: image (3) + masked_image (3) + mask (1)
                #concatenated = torch.cat([image, masked_image, mask], dim=1)  # [B, 7, H, W]
                
                # Update batch to match expected format
                batch = {
                    'image': concatenated,  # This is what the model expects
                    'txt': batch.get('txt', ''),  # Add text if available
                }
                
                # Debug: print on first batch
                if batch_idx == 0 and self.current_epoch == 0:
                    print(f"\n=== After preprocessing ===")
                    print(f"Concatenated input shape: {concatenated.shape}")
                    print(f"Expected: [batch_size, 7, height, width]")
                    print("================================\n")
            # === END FIX ===
            
            batch = {
                    'image': batch['image'],  # This is what the model expects
                    'masked_image': torch.cat([batch['masked_image'], batch['mask']],-1)
                }
            """
            # Forward pass - use training_step for LightningModule
            try:
                if hasattr(self.model, 'training_step'):
                    loss_dict = self.model.training_step(batch, batch_idx)
                else:
                    loss_dict = self.model(batch, batch_idx)
            except Exception as e:
                print(f"\nError in forward pass: {e}")
                print(f"Batch type: {type(batch)}")
                if isinstance(batch, dict):
                    print(f"Batch keys: {batch.keys()}")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                raise
            
            # ... rest of your training code

            
            if isinstance(loss_dict, dict):
                loss = loss_dict.get('loss', loss_dict.get('train/loss', list(loss_dict.values())[0]))
            else:
                loss = loss_dict
            
            # Normalize loss for gradient accumulation
            loss = loss / self.accumulate_grad_batches
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Logging
            epoch_loss += loss.item() * self.accumulate_grad_batches
            
            if self.global_step % self.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * self.accumulate_grad_batches:.4f}",
                    'lr': f"{lr:.2e}",
                    'step': self.global_step
                })
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def validate(self, dataloader):
        """Validation loop"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                try:
                    if hasattr(self.model, 'validation_step'):
                        loss_dict = self.model.validation_step(batch, batch_idx)
                    elif hasattr(self.model, 'training_step'):
                        loss_dict = self.model.training_step(batch, batch_idx)
                    else:
                        loss_dict = self.model(batch)
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
                
                if isinstance(loss_dict, dict):
                    loss = loss_dict.get('loss', loss_dict.get('val/loss', list(loss_dict.values())[0]))
                else:
                    loss = loss_dict
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss


    
    def fit(self, train_dataloader, num_epochs, val_dataloader=None):
        """Main training loop"""
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch(train_dataloader)
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Validation (if provided)
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.save_freq == 0:
                save_checkpoint(self.model, self.optimizer, epoch, self.global_step, 
                              self.logdir, f"epoch_{epoch:06d}.ckpt")
            
            # Always save last checkpoint
            save_checkpoint(self.model, self.optimizer, epoch, self.global_step, 
                          self.logdir, "last.ckpt")


if __name__ == "__main__":
    # Parse arguments
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    # Set up logging directory
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if opt.name:
        name = "_" + opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name = ""
    
    nowname = now + name + opt.postfix
    logdir = os.path.join(os.getcwd(), opt.logdir, nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    
    # Set seed
    seed_everything(opt.seed)
    
    # Load config
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    # Save config
    print("Config:")
    print(OmegaConf.to_yaml(config))
    OmegaConf.save(config, os.path.join(cfgdir, f"{now}-config.yaml"))
    
    # Device setup
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = instantiate_from_config(config.model)
    model = model.to(device)
    
    # Initialize optimizer
    base_lr = config.model.base_learning_rate
    batch_size = config.data.params.batch_size
    
    if opt.scale_lr:
        lr = opt.accumulate_grad_batches * batch_size * base_lr
        print(f"Scaled learning rate: {lr:.2e}")
    else:
        lr = base_lr
        print(f"Base learning rate: {lr:.2e}")
    
    # Use AdamW optimizer (common choice)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    if opt.resume:
        start_epoch, global_step = load_checkpoint(model, optimizer, opt.resume)
    
    # Set up data
    img_size = config.data.params.train.params.size
    path = config.data.params.train.target
    
    if opt.stage == "0":
        dataset = PIL_data.InpaintingTrain_autoencoder(img_size, path)
    elif opt.stage == "1":
        dataset = PIL_data.InpaintingTrain_ldm(img_size, path)
    else:
        raise ValueError(f"Unknown stage: {opt.stage}")
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=batch_size * 2,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(train_dataloader)}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        logdir=logdir,
        log_freq=opt.log_freq,
        save_freq=opt.save_freq,
        accumulate_grad_batches=opt.accumulate_grad_batches
    )
    
    trainer.global_step = global_step
    trainer.current_epoch = start_epoch
    
    # Training
    if opt.train:
        print("Starting training...")
        try:
            trainer.fit(train_dataloader, num_epochs=opt.epochs)
            print("Training completed!")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            save_checkpoint(model, optimizer, trainer.current_epoch, 
                          trainer.global_step, logdir, "interrupted.ckpt")
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            save_checkpoint(model, optimizer, trainer.current_epoch, 
                          trainer.global_step, logdir, "error.ckpt")
            raise
    
    print("Done!")
