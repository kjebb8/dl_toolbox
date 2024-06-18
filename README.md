# Deep Learning Toolbox
Small projects to demonstrate the key ideas, techniques and architectures of deep learning

## Highlights

### PyTorch Basics
- Tensors, gradients, regression and classification
- Neural network, CNN and RNN for MNIST classification

### Model Optimization
- CIFAR10 CNN classifier
- Hyperparameter tuning with RayTune
- Profile CPU, GPU time and memory with torch profiler
- Benchmarking average execution times, threads, algorithms, and instruction counts
- Efficient parameterization techniques for regularization
- Pruning for model compression

### PyTorch Tensorboard
- Tensorboard profiling visualization
- SummaryWriter for model and data visualization
- HTA as alternative for Tensorboard (Needs GPU, not tested)

### Variational Autoencoder
- Basic neural network implementation of Autoencoder and Variational Autoencoder
- MNIST dataset generation
- Latent space visualization

### GAN
- Basic GAN implementation and training on MNIST
- DCGAN with convolutional layers on CIFAR10 dataset (not trained to completion)

### Transformer
- Transformer implementation from scratch
- Transformer using PyTorch TransformerEncoder and TransformerDecoder
- Training on TinyShakespeare for generation
- NanoGPT model, training and benchmarking

### VQGAN
- VQGAN with Oxford flower dataset training and generation
- Swish activation
- Convolutional attention block (Non-local)
- Learned Codebook embedding for latent space
- Loss from discriminator, reconstruction error, and perceptual similarity (VGG-16)
- GPT transformer for learning sequences of codebook vectors and sampling for image generation

### Diffusion
- DDPM implementation with two UNet variations (double residual and attention residual)
- Denoising sampling of novel images
- Trained on MNIST and CIFAR (not to completion)