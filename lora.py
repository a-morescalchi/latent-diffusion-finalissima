import torch

import torch.nn as nn

from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock
import copy


    
class LowRankConv1d(nn.Module):
    def __init__(self, in_features, out_features, rank, zeros=False):
        super().__init__()
        self.zeros = zeros
        if not zeros: 
            self.lora_A = torch.nn.Conv1d(in_features,
                             rank,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            self.lora_B = torch.nn.Conv1d(rank,
                             out_features,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        
            nn.init.normal_(self.lora_A.weight, std=1/rank)
            nn.init.normal_(self.lora_B.weight, std=1/rank)

    def forward(self, x):
        if self.zeros:
            return torch.zeros_like(x)
        return self.lora_B( self.lora_A( x ))


class loraAttentionBlock(nn.Module):   
    def __init__(self, base_layer, rank=4, qkv=[True, True, True], alpha=1, dropout=0):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.qkv = qkv
        self.scaling = alpha / rank

        # Dimensions from the base layer
        self.channels = base_layer.channels

        # Initialize LoRA layers
        self.lora_q = LowRankConv1d(self.channels, self.channels, self.rank, zeros=not qkv[0])
        self.lora_k = LowRankConv1d(self.channels, self.channels, self.rank, zeros=not qkv[1])
        self.lora_v = LowRankConv1d(self.channels, self.channels, self.rank, zeros=not qkv[2])

        self.lora_proj_out = LowRankConv1d(self.channels, self.channels, self.rank)

    def forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)

        # 1. Normalization (Applied to BOTH paths)
        x_norm = self.base.norm(x_flat)
        
        # 2. Base Path (Frozen)
        base_qkv = self.base.qkv(x_norm)

        # 3. LoRA Path (Trainable)
        # FIX: Input is now x_norm, not x_flat
        lora_qkv = torch.cat([
            self.lora_q(x_norm), 
            self.lora_k(x_norm), 
            self.lora_v(x_norm)
        ], dim=1) * self.scaling

        # 4. Combine & Attend
        h = self.base.attention(base_qkv + lora_qkv)
        
        # 5. Output Projection
        # h is already the result of attention, so we just project it
        h_final = self.base.proj_out(h) + self.lora_proj_out(h) * self.scaling
        
        return (x_flat + h_final).reshape(b, c, *spatial)




class LoRAConv2dLayer(nn.Module):
    """
    Wraps a torch.nn.Conv2d layer.
    """
    def __init__(self, original_layer, rank=4, alpha=1):
        super().__init__()
        self.base = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        
        # Conv2d attributes
        in_channels = original_layer.in_channels
        out_channels = original_layer.out_channels
        kernel_size = original_layer.kernel_size
        stride = original_layer.stride
        padding = original_layer.padding
        
        # --- LoRA Decomposition for Conv2d ---
        # Matrix A: Down-project (In -> Rank). We keep the original kernel size/stride to match spatial reduction.
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        
        # Matrix B: Up-project (Rank -> Out). Kernel size is 1x1 to just mix channels.
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)

    def forward(self, x):
        # 1. Base (Frozen)
        base_out = self.base(x)
        
        # 2. LoRA
        # A maps (B, C_in, H, W) -> (B, Rank, H', W')
        # B maps (B, Rank, H', W') -> (B, C_out, H', W')
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        
        return base_out + lora_out
    


def apply_lora_to_layer(layer, rank=4, alpha=1, qkv=[True, False, True]):
    '''
    applies vera to a layer if it can
    '''
    new_layer = copy.deepcopy(layer)
    if isinstance(layer, AttentionBlock):
        new_layer = loraAttentionBlock(layer, rank=rank, qkv=qkv, alpha=alpha)

    return new_layer


#removed LinearAttention because not ready I am tired voglio morire
available_targets = [AttentionBlock]


def ignorant_lora(model, target_class=available_targets, rank=4, qkv=[True, False, True], alpha=1):
    """
    Applies Lora wherever it can in the model
    """
    #for name, module in model.named_modules():
    #    if isinstance(module, AttentionBlock):
    #        print(f"mashallah")

    #print([ name for name, module in model.named_modules() if any(isinstance(module, target) for target in target_class)])
    layers_to_replace = []
    for name, module in model.named_modules():
        for target in target_class: 
            if isinstance(module, target):
                layers_to_replace.append((name, module))

    for full_name, old_layer in layers_to_replace:

        if '.' in full_name:
            parent_name, child_name = full_name.rsplit('.', 1)

            parent_module = model.get_submodule(parent_name)
        else:
            parent_name = ""
            child_name = full_name
            parent_module = model

        #print(f"Replacing layer: {full_name}")
        new_layer = apply_lora_to_layer(old_layer, rank=rank, qkv=qkv, alpha=alpha)
        #print("New Parameters theoretical:", sum([len(n) for n, p in new_layer.named_parameters() if 'lora' in n]))
        setattr(parent_module, child_name, new_layer)
    return model


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")

# Inside your loraModel class (in lora.py)

def inject_lora(model, rank=4, alpha=1):
    """
    Injects LoRA into both AttentionBlocks and ResBlocks.
    """
    injected = 0
    for name, module in model.named_modules():
        
        # 1. Attention Blocks
        if isinstance(module, AttentionBlock):
            # We assume the user wants to wrap the whole block
            # Note: This replaces the module in-place if we find the parent.
            # But simpler approach: verify it hasn't been wrapped yet.
            pass # We rely on wrapping sub-components or the block itself. 
            # (Your previous logic wrapped the block. Let's stick to your loraAttentionBlock replacement logic)
            
    # Re-implementing the robust injection logic:
    layers_to_replace = []
    
    for name, module in model.named_modules():
        # Target 1: AttentionBlock
        if isinstance(module, AttentionBlock):
            if not isinstance(module, loraAttentionBlock):
                layers_to_replace.append((name, module, "Attention"))
        
        # Target 2: ResBlock (Convolutions)
        elif isinstance(module, ResBlock):
            # We inject directly into the sub-sequences of ResBlock
            # ResBlock has: .in_layers (Sequential), .out_layers (Sequential), .skip_connection (Conv2d)
            
            # Helper to replace last Conv2d in a Sequential
            def inject_into_sequential(seq):
                if len(seq) > 0 and isinstance(seq[-1], nn.Conv2d) and not isinstance(seq[-1], LoRAConv2dLayer):
                    seq[-1] = LoRAConv2dLayer(seq[-1], rank=rank, alpha=alpha)
                    return True
                return False

            if hasattr(module, "in_layers"):
                if inject_into_sequential(module.in_layers): injected += 1
            if hasattr(module, "out_layers"):
                if inject_into_sequential(module.out_layers): injected += 1
            if hasattr(module, "skip_connection") and isinstance(module.skip_connection, nn.Conv2d):
                 module.skip_connection = LoRAConv2dLayer(module.skip_connection, rank=rank, alpha=alpha)
                 injected += 1

    # Apply Attention replacements
    for full_name, old_layer, type_name in layers_to_replace:
        if '.' in full_name:
            parent_name, child_name = full_name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
        else:
            parent_name, child_name = "", full_name
            parent_module = model
            
        if type_name == "Attention":
            print(f"Injecting LoRA into Attention: {full_name}")
            new_layer = loraAttentionBlock(old_layer, rank=rank, alpha=alpha)
            setattr(parent_module, child_name, new_layer)
            injected += 1

    print(f"Total LoRA Layers Injected: {injected}")
    return model

def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total:.4f}")



class loraModel(nn.Module):
    def __init__(self, base_model, rank=4, alpha=1, qkv=[True, True, True]):
        super().__init__()
        # Use the new inject_lora function
        self.model = inject_lora(base_model, rank=rank, alpha=alpha)

    def forward(self, x, t=None, **kwargs):
        return self.model(x, t, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def set_trainable_parameters(self):
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print_trainable_parameters(self.model)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        full_sd = self.model.state_dict(destination, prefix, keep_vars)
        return {k: v for k, v in full_sd.items() if "lora" in k}