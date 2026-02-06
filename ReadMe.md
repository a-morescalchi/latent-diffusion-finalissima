
# Rectified Flow & Conditional OT for Inverse Problems

This repository implements a **Conditional Optimal Transport Flow Solver** for creation of deepfakes through **Rectified Flow** generation using Low-Rank Adaptation (LoRA) on Latent Diffusion Models (LDM).

## 1. Overview

We use a Low Rank Adapter on the diffusion model at https://github.com/CompVis/latent-diffusion, trained with the celeba-hq dataset. We train the LoRA on a dataset of photos of a specific individual. Using technioques from inverse problems we mark specific parts of the image to allow for a limited reconstruction resulting in a deepfake. 


## 2. Key Components

### `algo.py` (The Solver)

Contains the core mathematical framework based on *Theorem 1 (Conditional Vector Fields)*.

### `desperados.py` / Notebook (The Generator)

Handles the application fo the LoRA to the diffusion model.



## 3. Installation

**Prerequisites:** A GPU-enabled environment is highly recommended. 

If running on colab, the requirement below can be ignored.
**Install Dependencies**
```bash
pip -r requirements.txt

```

# Usage

!!! the repo must be renamed latent-diffusion once downloaded, otherwise the code might not work.

Just for visualizing results: 

Relevant notebooks: desperados.ipynb, despainting.py

The thing will become more systematic before this evening...

## 6. Citations

* *Rectified Flow / Flow Matching*: [Liu et al., "Flow Straight and Fast: Learning to Generate with Rectified Flow"]
* *Latent Diffusion*: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models".

