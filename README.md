# My Custom Stable Diffusion

This project is a customized version of the official [Stable Diffusion](https://github.com/CompVis/stable-diffusion).  
I have added new features and modified some internal components for my own use cases (e.g. custom generation pipeline, additional models, or integration with [your code name here]).

Original repo: [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)

## Major Changes
- ✨ Latent code evolution with CMA-ES
- 🤖 CLIP-guided fitness evaluation
- 🔄 Multi-objective prompt optimization


# 🧬 Prompt-Aligned Latent Optimization with Codebook-Guided CMA-ES

This project implements an evolutionary optimization pipeline to generate high-quality, prompt-aligned images using Stable Diffusion. It combines prompt clustering, a latent codebook, CLIP-based fitness evaluation, and CMA-ES to iteratively evolve latent codes toward optimal image generation.

---

## 📌 Features

- 🧠 **Prompt Clustering with MSCOCO**  
  Clustered the MSCOCO dataset prompts using sentence embeddings to build a prompt-aware **latent codebook**.

- 🧬 **Latent Initialization via Codebook**  
  For any given input prompt, the system searches for the **most similar prompt cluster** in the codebook and retrieves the corresponding latent code as initialization for evolution.

- 🔁 **Multi-Tasking Evolutionary Optimization**
Multiple prompts are optimized simultaneously. Each prompt corresponds to an individual CMA-ES population, and these populations share knowledge through:

  - Latent code sharing between related prompts.
  - Cross-task mutation and crossover strategies.

- 🎯 **Multi-Objective Fitness Function**  
  Image quality is evaluated using a combination of:
  - **CLIP Score**: Measures semantic alignment between the prompt and generated image.
  - **Photorealism Estimation**: Encourages realistic-looking outputs (optional; plug-in compatible).

- ⚙️ **Evolution Strategy: CMA-ES**  
  Latent codes in Stable Diffusion's latent space (e.g., `4×64×64`) are evolved using **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** for efficient high-dimensional optimization.

---

## 🚀 Pipeline Overview

1. **Prompt Clustering & Codebook Generation**
   - Cluster prompts in MSCOCO using sentence embeddings.
   - For each cluster, generate and store a representative latent code.

2. **Prompt Input & Latent Retrieval**
   - Given a new input prompt, compute similarity against codebook clusters.
   - Use the most similar latent code as initialization for CMA-ES.

3. **Latent Evolution (CMA-ES)**
   - Optimize the latent code over several generations.
   - Each generation evaluates samples via CLIP-based fitness.

4. **Image Synthesis**
   - The best latent is decoded by Stable Diffusion to produce the final image.

---

## 🛠️ Technologies Used

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Latent space image generation
- [CLIP](https://github.com/openai/CLIP) - Semantic similarity scoring
- [CMA-ES](https://github.com/CyberAgentAILab/cmaes) - Evolution Strategy optimizer
- [Sentence Transformers](https://www.sbert.net/) - Prompt embedding & clustering
- [MSCOCO](https://cocodataset.org/) - Dataset used for codebook training

---

## ⚙️ Usage

This project uses the same command-line arguments and usage as the original Stable Diffusion script.

For full usage instructions and CLI options, please refer to the [official Stable Diffusion repository](https://github.com/CompVis/stable-diffusion).

