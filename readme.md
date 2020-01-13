# Solar Magnetogram Denoising (TensorFlow2)

Park et al., 2019, ApJL, submitted

Title: Denoising SDO/HMI Solar Magnetogram by Deep Learning based on Generative Adversarial Network

## Network Architectures

The architectures are based on pix2pix (Isola+ 2016) and https://github.com/eunsu-park/solar_euv_generation (Park+ 2019, ApJL)

Discriminator: PatchGAN discriminator

Generator: modified UNet generator

## To run this code,

GAN: change some parameters in denoising_gan.ipynb (especially first cell), and run it.
