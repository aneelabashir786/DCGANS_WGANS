# app.py
import streamlit as st
import torch
import numpy as np
import os
from utils import load_model, generate_image, generate_grid, clear_model_cache, MODEL_URLS

# Page configuration
st.set_page_config(
    page_title="Anime Face Generator",
    page_icon="",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        font-size: 1.1em;
        font-weight: bold;
    }
    .success-message {
        padding: 1em;
        border-radius: 0.5em;
        background-color: #d4edda;
        color: #155724;
        margin: 1em 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and header
st.title(" Anime Face Generator with GANs")
st.markdown("""
    <div style='text-align: center; margin-bottom: 2em;'>
        <p>Generate unique anime-style faces using DCGAN and WGAN-GP models</p>
        <p><strong>Models hosted on Hugging Face</strong></p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header(" Configuration")
    
    # Model sources info
    with st.expander(" Model Sources"):
        st.markdown("**DCGAN Model:**")
        st.code(MODEL_URLS["DCGAN"], language="text")
        st.markdown("**WGAN-GP Model:**")
        st.code(MODEL_URLS["WGAN-GP"], language="text")
        st.info("Models are downloaded automatically from Hugging Face on first use")
        
        if st.button(" Clear Model Cache", help="Delete downloaded models to re-download"):
            if clear_model_cache():
                st.success("Cache cleared! Models will be re-downloaded on next load")
                st.rerun()
    
    st.divider()
    
    # Generation mode
    generation_mode = st.radio(
        " Mode",
        ["Single Model", "Compare Both"],
        help="Single: Use one model\nCompare: See both side by side",
        horizontal=True
    )
    
    if generation_mode == "Single Model":
        model_type = st.selectbox(
            "Select Model",
            ["DCGAN", "WGAN-GP"],
            help="DCGAN: Faster generation\nWGAN-GP: Better quality"
        )
    
    # Generation parameters
    st.subheader(" Generation Settings")
    n_images = st.slider(
        "Number of images",
        min_value=1,
        max_value=64,
        value=9,
        help="How many images to generate"
    )
    
    use_seed = st.checkbox(" Use fixed random seed", help="Get reproducible results")
    if use_seed:
        seed = st.number_input("Seed value", min_value=0, max_value=9999, value=42)
    else:
        seed = None
    
    # Device selection (auto-detect)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.caption(f"🖥️ Device: {device.upper()}")
    
    st.divider()
    
    # Load model button
    if generation_mode == "Single Model":
        if st.button(f" Load {model_type} Model", type="primary", use_container_width=True):
            with st.spinner(f"Loading {model_type} model from Hugging Face..."):
                model = load_model(model_type, device)
                if model is not None:
                    st.session_state['model'] = model
                    st.session_state['model_type'] = model_type
                    st.session_state['loaded'] = True
                    st.success(f" {model_type} model loaded successfully!")
                else:
                    st.error("Failed to load model. Please check your internet connection.")
    else:
        if st.button(" Load Both Models", type="primary", use_container_width=True):
            # Load DCGAN
            with st.spinner("Loading DCGAN model from Hugging Face..."):
                dcgan_model = load_model("DCGAN", device)
                if dcgan_model is not None:
                    st.session_state['dcgan'] = dcgan_model
                    st.success(" DCGAN model loaded!")
                else:
                    st.error("Failed to load DCGAN model")
            
            # Load WGAN-GP
            with st.spinner("Loading WGAN-GP model from Hugging Face..."):
                wgan_model = load_model("WGAN-GP", device)
                if wgan_model is not None:
                    st.session_state['wgan'] = wgan_model
                    st.success(" WGAN-GP model loaded!")
                else:
                    st.error("Failed to load WGAN-GP model")
            
            if 'dcgan' in st.session_state and 'wgan' in st.session_state:
                st.session_state['both_loaded'] = True
                st.balloons()

# Main content area
st.divider()

if generation_mode == "Single Model":
    # Single model generation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button(" Generate Images", type="primary", use_container_width=True):
            if 'loaded' not in st.session_state:
                st.error(" Please load a model first!")
            else:
                with st.spinner(f"Generating {n_images} image(s) with {st.session_state['model_type']}..."):
                    if n_images == 1:
                        img = generate_image(
                            st.session_state['model'], 
                            device=device, 
                            seed=seed
                        )
                        st.image(img, caption=f"{st.session_state['model_type']} - Generated Face", use_container_width=True)
                    else:
                        grid_img = generate_grid(
                            st.session_state['model'], 
                            n_images=n_images, 
                            device=device, 
                            seed=seed
                        )
                        st.image(grid_img, caption=f"{st.session_state['model_type']} - {n_images} Generated Faces", use_container_width=True)
                    
                    st.success(" Generation complete!")
        
        if 'loaded' not in st.session_state:
            st.info(" Please load a model from the sidebar to start generating faces!")

else:  # Comparison Mode
    st.header(" Model Comparison")
    st.markdown("Compare the quality and style of images generated by DCGAN and WGAN-GP")
    
    # Comparison settings
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Option to use same seed for fair comparison
        use_same_seed = st.checkbox("Use same random seed for fair comparison", value=True)
        
        if st.button(" Compare Models", type="primary", use_container_width=True):
            if 'both_loaded' not in st.session_state:
                st.error(" Please load both models first!")
            else:
                # Determine seeds
                if use_same_seed:
                    compare_seed = seed if seed is not None else 42
                else:
                    compare_seed = None
                
                with st.spinner("Generating comparison images..."):
                    # Create two columns for side-by-side comparison
                    col1, col2 = st.columns(2, gap="large")
                    
                    with col1:
                        st.markdown("###  DCGAN")
                        st.caption("*Standard GAN - Faster generation*")
                        
                        if n_images == 1:
                            dcgan_img = generate_image(
                                st.session_state['dcgan'], 
                                device=device, 
                                seed=compare_seed
                            )
                            st.image(dcgan_img, caption="DCGAN Generated Face", use_container_width=True)
                        else:
                            dcgan_grid = generate_grid(
                                st.session_state['dcgan'], 
                                n_images=n_images, 
                                device=device, 
                                seed=compare_seed
                            )
                            st.image(dcgan_grid, caption=f"DCGAN - {n_images} Faces", use_container_width=True)
                        
                        # DCGAN metrics
                        st.info("**DCGAN Features:**\n Fast generation\n Simple architecture\n May have mode collapse")
                    
                    with col2:
                        st.markdown("###  WGAN-GP")
                        st.caption("*Wasserstein GAN - Better quality*")
                        
                        if n_images == 1:
                            wgan_img = generate_image(
                                st.session_state['wgan'], 
                                device=device, 
                                seed=compare_seed
                            )
                            st.image(wgan_img, caption="WGAN-GP Generated Face", use_container_width=True)
                        else:
                            wgan_grid = generate_grid(
                                st.session_state['wgan'], 
                                n_images=n_images, 
                                device=device, 
                                seed=compare_seed
                            )
                            st.image(wgan_grid, caption=f"WGAN-GP - {n_images} Faces", use_container_width=True)
                        
                        # WGAN-GP metrics
                        st.success("**WGAN-GP Features:**\n Higher quality\n More diverse\n Stable training\n Slower generation")
                    
                    st.success(" Comparison complete!")
        
        if 'both_loaded' not in st.session_state:
            st.info(" Please load both models from the sidebar to start comparison!")

# Quick demo section
st.divider()
st.header(" Quick Demo")

demo_col1, demo_col2, demo_col3 = st.columns(3)

with demo_col1:
    if st.button(" Quick DCGAN Sample", use_container_width=True):
        if 'dcgan' in st.session_state or ('loaded' in st.session_state and st.session_state.get('model_type') == "DCGAN"):
            model = st.session_state.get('dcgan', st.session_state.get('model'))
            with st.spinner("Generating DCGAN sample..."):
                sample = generate_grid(model, n_images=9, device=device)
                st.image(sample, caption="DCGAN Sample (9 images)", use_container_width=True)
        else:
            st.warning("Please load DCGAN model first!")

with demo_col2:
    if st.button(" Quick WGAN-GP Sample", use_container_width=True):
        if 'wgan' in st.session_state or ('loaded' in st.session_state and st.session_state.get('model_type') == "WGAN-GP"):
            model = st.session_state.get('wgan', st.session_state.get('model'))
            with st.spinner("Generating WGAN-GP sample..."):
                sample = generate_grid(model, n_images=9, device=device)
                st.image(sample, caption="WGAN-GP Sample (9 images)", use_container_width=True)
        else:
            st.warning("Please load WGAN-GP model first!")

with demo_col3:
    st.markdown("###  Model Stats")
    st.metric("Model Size (DCGAN)", "~50 MB")
    st.metric("Model Size (WGAN)", "~50 MB")

# Information section
with st.expander(" About Models & Training"):
    st.markdown("""
    ### Model Architectures
    
    **DCGAN (Deep Convolutional GAN)**
    - Generator: 5 transposed convolution layers
    - Batch Normalization for stable training
    - ReLU activations in generator
    - Trained for 50 epochs
    
    **WGAN-GP (Wasserstein GAN with Gradient Penalty)**
    - Same generator architecture as DCGAN
    - Instance Normalization instead of BatchNorm
    - Wasserstein loss with gradient penalty
    - Trained for 60 epochs
    
    ### Training Details
    - **Dataset**: Anime faces dataset
    - **Image Size**: 64×64 pixels
    - **Batch Size**: 64
    - **Learning Rate**: 0.0002
    - **Optimizer**: Adam (β1=0.5, β2=0.999)
    """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; padding: 1em;'>
        <p>Built with  using PyTorch, Streamlit, and  Hugging Face</p>
        <p>Models trained on anime face dataset | <a href='https://github.com/YOUR_USERNAME/gan-face-generator' target='_blank'>GitHub Repository</a></p>
    </div>
""", unsafe_allow_html=True)
