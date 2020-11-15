# Solar Magnetogram Denoising (TensorFlow2)

Park et al., 2020, ApJL, 891, L4, https://doi.org/10.3847/2041-8213/ab74d2

Title: De-noising SDO/HMI Solar Magnetograms by Image Translation Method Based on Deep Learning

## Network Architectures

### Generative Adversarial Network

The architectures are based on pix2pix (Isola+ 2016) and https://github.com/eunsu-park/solar_euv_generation (Park+ 2019, ApJL)

Discriminator: PatchGAN discriminator

Generator: modified UNet generator

### Auto Encoder

Generator: modified UNet generator

## To run this code,

Change some parameters in denoising_gan.ipynb (GAN) or denoising_ae.ipylab (AE), especially first cell, and run it.
