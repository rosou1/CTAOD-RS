<p align="center">
  <h1 align="center">
    Continual Test-Time Adaptation for Remote Sensing Object Detection
  </h1>
  
  <p align="center">
            Mingjing Yong,
      Nanqing Liu,
      Sen Lei,
      Jie Pan,
      Xue Yang,
      and
      Heng-Chao Li
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

Official PyTorch implementation of  **Continual Test-Time Adaptation for Remote Sensing Object Detection**

<div align="center">
  <img src="overview.png" width="100%"/>
</div>


---

## ✅ TODO
- [x] Initialize repository structure
- [ ] Release paper link
- [ ] Release pretrained checkpoints

---

## 📖 Introduction

Remote sensing object detection (RSOD) plays a crucial role in applications such as urban planning, disaster response, and environmental monitoring. Despite advances in generic object detection, RSOD remains challenging due to complex variations across sensors, geographic regions, illumination, weather, and object scales, which induce severe domain shifts when models trained on source domains are deployed on unseen target domains. Test-time adaptation (TTA) offers a practical solution by enabling online model adaptation using only incoming test data, but conventional TTA methods generally assume relatively static target distributions, limiting their effectiveness under the dynamic and diverse conditions of RSOD. Continual test-time adaptation for object detection (CTAOD) extends TTA to sequentially arriving test samples, yet existing approaches often rely on global feature updates with fixed thresholds, leading to noisy feature propagation, error accumulation, and catastrophic forgetting. In this paper, we propose CTAOD-RS, a novel framework for continual test-time adaptive object detection in remote sensing imagery. The proposed framework leverages Spectral Routing to decompose features into frequency and spatial components, suppressing noise and clutter, and employs Historical Reconstruction to dynamically determine when model adaptation is necessary, avoiding unnecessary updates and mitigating catastrophic forgetting. Extensive experiments on benchmark RSOD datasets, including the newly introduced DIOR-C corrupted benchmark designed to simulate realistic remote sensing domain shifts, demonstrate superior detection accuracy and robustness compared to existing TTA and CTAOD methods, while maintaining high efficiency. 


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
```
---
## 📂 Dataset Preparation

- Download the DIOR-C at the |[Baidu Cloud](https://pan.baidu.com/s/1g01ATN473z3kfr73w88R-A?pwd=rwvf)|
- Download the source features statistics of DIOR at the |[Baidu Cloud](https://pan.baidu.com/s/1pMDURPX_tfUDo1G-T8xLsw?pwd=sv8n)|
---
## 🔧 Quick Start / How to Run

### 1. Collect Source Feature Statistics

Before running test-time adaptation, you need to collect feature statistics from the source domain (DIOR-clean). These statistics capture the source feature distribution (mean and covariance) used for KL divergence alignment during adaptation.

```bash
# ResNet50 backbone
python tools/train_net.py \
    --config-file configs/Base/COCO_faster_rcnn_R50_FPN_1x.yaml \
    --eval-only \
    MODEL.WEIGHTS /path/to/r50_weights.pth \
    TEST.CONTINUAL_DOMAIN False \
    TEST.COLLECT_FEATURES True \
    OUTPUT_DIR outputs/stats/r50

# SwinT backbone
python tools/train_net.py \
    --config-file configs/Base/COCO_faster_rcnn_swinT_FPN_1x.yaml \
    --eval-only \
    MODEL.WEIGHTS /path/to/swt_weights.pth \
    TEST.CONTINUAL_DOMAIN False \
    TEST.COLLECT_FEATURES True \
    OUTPUT_DIR outputs/stats/swt
```

### 2. Run Test-Time Adaptation

**SpectralAdapter (Frequency-Domain Routing):**

```bash
# ResNet50 backbone
python tools/train_net.py \
    --config-file configs/TTA/COCO_FFTR50.yaml \
    --eval-only \
    TEST.ADAPTATION.WHERE adapter \
    OUTPUT_DIR outputs/tta/fftr50

# SwinT backbone
python tools/train_net.py \
    --config-file configs/TTA/COCO_FFTswinT.yaml \
    --eval-only \
    TEST.ADAPTATION.WHERE adapter \
    OUTPUT_DIR outputs/tta/fftswt
```

**SpectralAdapter + Historical Reconstruction (VAE-based):**

```bash
# ResNet50 backbone
python tools/train_net.py \
    --config-file configs/TTA/COCO_R50recon.yaml \
    --eval-only \
    TEST.ADAPTATION.WHERE adapter \
    OUTPUT_DIR outputs/tta/r50recon

# SwinT backbone
python tools/train_net.py \
    --config-file configs/TTA/COCO_swinTrecon.yaml \
    --eval-only \
    TEST.ADAPTATION.WHERE adapter \
    OUTPUT_DIR outputs/tta/swtrecon
```

### 3. (Optional) Enable WandB Logging

Add `--wandb` to any command above to log metrics to Weights & Biases.
---
## 📝 License

- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---
## 🙌 Acknowledgments

- Our work is inspired by WHW(https://github.com/natureyoo/ContinualTTA_ObjectDetection) and Earth-Adapter(https://github.com/VisionXLab/Earth-Adapter). We are grateful for their outstanding work and code.
---
## 📚 Citation

If you find this work helpful in your research, please consider citing:

```bibtex
@article{yong2026continual,
  title={Continual Test-Time Adaptive Object Detection in Remote Sensing Images via Spectral Routing and Historical Reconstruction},
  author={Yong, Mingjing and Liu, Nanqing and Lei, Sen and Pan, Jie and Yang, Xue and Li, Heng-Chao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```