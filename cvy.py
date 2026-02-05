import torch
import sys

def convert_checkpoint(input_path, output_path):
    """Convert old checkpoint to new format"""
    print(f"Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Add missing keys
    if 'pytorch-lightning_version' not in checkpoint:
        checkpoint['pytorch-lightning_version'] = '1.9.0'
    
    if 'epoch' not in checkpoint:
        checkpoint['epoch'] = 0
    
    if 'global_step' not in checkpoint:
        checkpoint['global_step'] = 0
    
    # Ensure state_dict exists
    if 'state_dict' not in checkpoint:
        print("Warning: No state_dict found in checkpoint")
        checkpoint['state_dict'] = checkpoint  # Assume entire checkpoint is state_dict
    
    print(f"Saving converted checkpoint to {output_path}")
    torch.save(checkpoint, output_path)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py <input.ckpt> <output.ckpt>")
        sys.exit(1)
    
    convert_checkpoint(sys.argv[1], sys.argv[2])
