<div align="center" style="font-family: charter;">

<img src="assets/title.png" width="100%"/>

<br />

<div align="center">
<a href="https://github.com/linkangheng/Open-Vision-Reasoner/tree/main/paper/Open-Vision-Reasoner.pdf" target="blank" style="margin-right: 10px;">
    <img alt="arXiv" src="https://img.shields.io/badge/Paper-OVR-red?logo=arxiv" height="20" />
</a><a href="https://huggingface.co/collections/Kangheng/ovr-686646849f9b43daccbe2fe0" target="blank" style="margin-right: 10px;">
    <img alt="HF Model: OVR" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-OVR-fb8740?&logoColor=white" height="20" />
</a><a href="" target="blank" style="margin-right: 10px;">
    <img alt="Dataset: OVR" src="https://img.shields.io/badge/%F0%9F%97%84%EF%B8%8F%20_Dataset(coming)-OVR-48b9d0?&logoColor=white" height="20" />
</a>
</div>
</div>

## 📖 Overview

<img src="assets/preview.png" width="100%"/>

The remarkable reasoning capbility of Large Language Models (LLMs) stems from cognitive behaviors that emerge when reinforcing against verifiable rewards. This work investigates how to transfer this principle to Multimodal LLMs (MLLMs) to unlock **advanced visual reasoning**.

We introduce a two-stage paradigm built on Qwen2.5-VL-7B: a massive **text-only cold-start fine-tuning**, followed by **multimodal reinforcement learning** (RL) spanning nearly 1,000 steps—surpassing all prior open-source efforts in scale. This pioneering work reveals three fundamental insights: 

1. Behavior transfer emerges surprisingly early in cold start due to linguistic mental imagery. 
2. Cold start broadly memorizes visual behaviors, while RL critically discerns and scales up effective patterns. 
3. Transfer strategically favors high-utility behaviors such as visual reflection. 

Our resulting model, Open-Vision-Reasoner (OVR), achieves state-of-the-art performance on a suite of reasoning benchmarks, including **95.3%** on MATH500, **51.8%** on MathVision and **54.6%** on MathVerse. We release our model, data, and training dynamics to catalyze the development of more capable, behavior-aligned multimodal reasoners.

## 🚀 Model Release

> Models are available at Huggingface Collections: [Open-Vision-Reasoner](https://huggingface.co/collections/Kangheng/ovr-686646849f9b43daccbe2fe0). We release the intermediate cold-start model and the final RL-tuned OVR model to facilitate further research.

| **Model** | **Description** | **Download** |
|:---------:|:---------------:|:------------:|
| OVR-7B-ColdStart | Intermediate model after massive language-only cold-start fine-tuning | [🤗 OVR-7B-ColdStart](https://huggingface.co/Kangheng/OVR-7B-ColdStart) |
| OVR-7B-RL | Final model after large-scale multimodal RL training | [🤗 OVR-7B-RL](https://huggingface.co/Kangheng/OVR-7B-RL) |

## 📊 Performance Results

### **Language Reasoning**

The initial cold-start phase equips OVR with formidable language reasoning capabilities, outperforming all open-source 7B models on key math and logic benchmarks.

<p align="center">
  <img width="95%" src="assets/language_benchmarks.png">
</p>

### **Visual Reasoning**

Crucially, these linguistic skills translate into state-of-the-art performance on visual reasoning tasks, validating the effectiveness of our cognitive transfer approach.

<p align="center">
  <img width="95%" src="assets/visual_benchmarks.png">
</p>

### **Cognitive Behavior Analysis**

Our analysis systematically tracks the emergence of cognitive behaviors, confirming that they are learned during the cold-start and amplified by RL.

<p align="center">
  <img width="45%" src="assets/transfer-a.png">
  <img width="46%" src="assets/transfer-b.png">
</p>

> [!IMPORTANT]
> Linguistic cognitive patterns, once established, can be powerfully transferred and amplified for visual reasoning through targeted reinforcement learning.

## 🔧 Training Pipeline

To facilitate efficient cognitive development and cross-modal generalization, we employ the popular "RL with a cold start" paradigm with two sequential training stages:

**Stage 1: Linguistic Cold Start**  
The LLM module is supervised fine-tuned on language-only reasoning datasets distilled from DeepSeek-R1, establishing core cognitive behaviors such as backtracking and subgoal decomposition within a purely linguistic setting.

**Stage 2: Multimodal RL**
We apply reinforcement learning with Open-Reasoner-Zero setting on both text and multimodal tasks using verifiable match rewards. This promotes reasoning generalization and aligns previously learned cognitive patterns with visual contexts, enabling effective cross-modal transfer.

## 📊 Training Dynamics and Performance Evolution

<p align="center">
  <img width="50%" src="assets/coldstart_dynamic.png">
  <img width="50%" src="assets/rl_dynamics.png">
</p>

<p align="center">
  <img width="100%" src="assets/performance.png">
</p>

## 📋 Roadmap

- [x] `2025-07-04` 🎄: Initial release of OVR models, and research paper.
- [ ] 📊: Release of OVR training data.
- [ ] 🚀: Continuously iterate on models and data to release more powerful versions of OVR. Stay tuned!

## 📚 Citation

If you find our work useful for your research, please consider citing our paper:
```bibtex

```

