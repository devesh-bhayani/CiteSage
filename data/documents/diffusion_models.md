# Denoising Diffusion Probabilistic Models

Diffusion models are a class of generative model that learn to produce data (most commonly images, though the approach generalizes to audio, video, and other modalities) by learning to reverse a gradual noising process, rather than generating a sample in a single forward pass the way a GAN generator or a VAE decoder does.

## The Forward (Noising) Process

The forward process is a fixed, non-learned Markov chain that gradually adds a small amount of Gaussian noise to a real data sample x_0 over a sequence of T time steps, producing progressively noisier versions x_1, x_2, ..., x_T. At each step:

x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon,  where epsilon ~ N(0, I)

Here beta_t is a small variance value from a fixed noise schedule (beta_t typically increases across the T steps). After enough steps, x_T is (by construction) indistinguishable from pure Gaussian noise, regardless of what the original data sample x_0 was. Because each step is a simple Gaussian transition, there is a closed-form expression that lets x_t be sampled directly from x_0 in a single step (without simulating every intermediate step individually), which makes training efficient — training doesn't require actually running the full T-step chain forward for every example.

## The Reverse (Denoising) Process

Generation works by learning to reverse this process: starting from pure noise x_T, a neural network is used to iteratively predict and remove a small amount of noise at each step, gradually transforming x_T back toward a realistic data sample x_0. Since the exact reverse of the true forward process is intractable to compute directly, a neural network — typically a U-Net architecture with convolutional downsampling and upsampling paths connected by skip connections, often augmented with self-attention layers at lower spatial resolutions — is trained to approximate it, usually by predicting the noise component epsilon that was added at a given step t, given the noisy sample x_t and the timestep t itself.

## Training Objective

The standard training objective for a denoising diffusion model is surprisingly simple given the mathematical complexity underlying the model: at each training step, a random timestep t and a random real data sample x_0 are chosen, Gaussian noise epsilon is sampled and used to construct the corresponding noisy sample x_t via the closed-form forward-process expression, and the network is trained to predict that noise epsilon from x_t and t, using a straightforward mean-squared-error loss between the predicted and actual noise:

L = E[|| epsilon - epsilon_theta(x_t, t) ||^2]

This reframes the difficult problem of generative modeling as a much simpler, iterative noise-prediction regression problem, which turns out to train very stably compared to the adversarial training used by GANs.

## Sampling

Once trained, generating a new sample means starting from x_T sampled as pure Gaussian noise, and repeatedly applying the learned reverse step — using the network's noise prediction to estimate and remove a portion of the noise at each timestep — until arriving at x_0, a full-resolution generated sample. Because this typically requires many sequential network evaluations (the original formulation used on the order of a thousand steps), naive diffusion sampling is substantially slower than a single forward pass through a GAN generator; a significant body of follow-up work (including DDIM and various distillation techniques) has focused specifically on reducing the number of sampling steps required, in some cases to just a handful, while preserving generation quality.

## Conditioning and Text-to-Image Generation

Diffusion models can be conditioned on additional information — most notably, a text description — by injecting a text embedding (commonly produced by a pretrained language-vision model) into the denoising network at each step, typically via cross-attention layers within the U-Net, analogous to how the original Transformer's decoder attends to encoder outputs. This conditioning is the basis for text-to-image diffusion systems, which generate an image by running the reverse denoising process while steering every step toward a sample consistent with the provided text description.
