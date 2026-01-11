<p align="center">
  <h1 align="center">
    Continual Test-Time Adaptation for Remote Sensing Object Detection
  </h1>
  
  <p align="center">
      Derrick<sup></sup>&emsp;
      <!-- 可后续补充作者 -->
      <br>
      <!-- <h3 align='center'>CVPR 2026</h3> -->
      <div align="center">
      <!-- <a href='https://arxiv.org/abs/XXXX.XXXXX'><img src='https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arxiv'></a> -->
      <!-- <a href='https://github.com/yourname/yourrepo'><img src='https://img.shields.io/badge/GitHub-Code-yellow.svg?logo=github'></a> -->
      </div>
    <p align='center'>
        If you find our work helpful, please consider giving us a ⭐!
    </p>
   </p>
</p>

<!--
[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
-->

Official PyTorch implementation of  
**Continual Test-Time Adaptation for Remote Sensing Object Detection**

<div align="center">
  <img src="overview.jpg" width="100%"/>
</div>


---

## 🔔 Notice
This repository is under active development and organization.  
More code, checkpoints, and instructions will be released soon.

If you encounter any issues, feel free to open an issue.

---

## ✅ TODO
- [x] Initialize repository structure
- [ ] Release paper link
- [ ] Release pretrained checkpoints

---

## 📖 Introduction

This repository contains the official implementation of our work on  
**Continual Test-Time Adaptation (CTTA)** for **Remote Sensing Object Detection (RSOD)**.

Unlike conventional **Unsupervised Domain Adaptation (UDA)** or static
**Test-Time Adaptation (TTA)** settings, this work focuses on a more realistic
**continual test-time adaptation scenario**, where the target domain evolves
dynamically over time and ground-truth labels are unavailable.

We propose a robust adaptation framework that enables stable and reliable
online adaptation under long-term domain shift in remote sensing imagery.

---

## 📄 Paper

**Title:** Continual Test-Time Adaptation for Remote Sensing Object Detection  
**Authors:** Derrick et al.  
**Status:** Under review  



---

## 🛠️ Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.3 (recommended)
- Other dependencies listed in `requirements.txt`

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourname/yourrepo.git
cd yourrepo

# Create conda environment
conda create -n cttaod python=3.9 -y
conda activate cttaod

# Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

## 📂 Dataset Preparation


- Download the DIOR-C at the |[Baidu Cloud](https://pan.baidu.com/s/1WGoVqLuJTJXc2AVDyBxXYQ?pwd=s6rk)|

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
