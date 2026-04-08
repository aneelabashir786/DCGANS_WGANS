# utils.py
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import torchvision
import os
import requests
from tqdm import tqdm

# Model architectures
class DCGAN_Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(ngf*8), 
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ngf*4), 
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ngf*2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ngf), 
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), 
            nn.Tanh()
        )
    
    def forward(self, z): 
        return self.main(z)

class WGAN_Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(ngf*8), 
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ngf*4), 
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ngf*2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ngf), 
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), 
            nn.Tanh()
        )
    
    def forward(self, z): 
        return self.main(z)

# Hugging Face model URLs
MODEL_URLS = {
    "DCGAN": "https://huggingface.co/aneelaBashir22f3414/Tackling_Mode_Collapse_in_GANS/resolve/main/dcgan_G_ep50.pth",
    "WGAN-GP": "https://huggingface.co/aneelaBashir22f3414/Tackling_Mode_Collapse_in_GANS/resolve/main/wgan_G_ep60.pth"
}

def download_model_from_hf(model_type, cache_dir="./models"):
    """Download model from Hugging Face if not already cached"""
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get the URL for the model
    url = MODEL_URLS[model_type]
    
    # Extract filename from URL
    filename = url.split("/")[-1]
    local_path = os.path.join(cache_dir, filename)
    
    # Check if model already exists
    if os.path.exists(local_path):
        st.info(f"📁 Using cached {model_type} model")
        return local_path
    
    # Download the model
    with st.spinner(f"📥 Downloading {model_type} model from Hugging Face..."):
        try:
            # Stream download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Download with progress tracking
            with open(local_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloading {filename}: {progress*100:.1f}%")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ {model_type} model downloaded successfully!")
            return local_path
            
        except Exception as e:
            st.error(f"❌ Error downloading {model_type} model: {str(e)}")
            return None

@st.cache_resource
def load_model(model_type, device='cpu'):
    """Load trained generator model from Hugging Face"""
    
    # Download model from Hugging Face
    model_path = download_model_from_hf(model_type)
    
    if model_path is None:
        return None
    
    # Create model instance
    if model_type == 'DCGAN':
        model = DCGAN_Generator()
    else:  # WGAN-GP
        model = WGAN_Generator()
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Remove 'module.' prefix if present (from DataParallel training)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        # Load the modified state dict
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def generate_image(model, nz=100, device='cpu', seed=None):
    """Generate a single image from random noise"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_img = model(noise)
        # Denormalize from [-1, 1] to [0, 1]
        img = (fake_img.cpu() * 0.5 + 0.5).clamp(0, 1)
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        return (img * 255).astype(np.uint8)

def generate_grid(model, n_images=64, nz=100, device='cpu', seed=None):
    """Generate a grid of images"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        noise = torch.randn(n_images, nz, 1, 1, device=device)
        fake_imgs = model(noise)
        # Denormalize
        fake_imgs = (fake_imgs.cpu() * 0.5 + 0.5).clamp(0, 1)
        
        # Create grid
        grid_size = int(np.sqrt(n_images))
        grid_img = torchvision.utils.make_grid(fake_imgs, nrow=grid_size, padding=2, normalize=False)
        grid_img = grid_img.permute(1, 2, 0).numpy()
        return (grid_img * 255).astype(np.uint8)

def clear_model_cache():
    """Clear downloaded model cache"""
    import shutil
    cache_dir = "./models"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        return True
    return False
