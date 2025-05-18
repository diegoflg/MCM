# MCM: Multi-label Recognition under Noisy Supervision — A Confusion Mixture Modeling Approach

[ICASSP 2025]  
This repository contains the official PyTorch implementation of our paper.
DOI: [10.1109/ICASSP49660.2025.10889155](https://doi.org/10.1109/ICASSP49660.2025.10889155)

---

## Prerequisites

Environment requirements:

- Python: 3.8.8  
- PyTorch: 1.10.0  
- Torchvision: 0.11.1  
- CUDA: 11.3  
- cuDNN: 8200  
- GPU: NVIDIA A100-SXM4-40GB  
- NumPy: 1.21.2  
- Pandas: 1.2.4

To install dependencies, use:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download the PASCAL VOC 2007 and 2012 datasets using the following commands:

```bash
# VOC 2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar && tar -xvf VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar -xvf VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar && tar -xvf VOCtestnoimgs_06-Nov-2007.tar

# VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar && tar -xvf VOCdevkit_18-May-2011.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar && tar -xvf VOCtrainval_11-May-2012.tar
```

Place all extracted data into a root folder and note the path for training.

---

## Running the Model

To train the model on the VOC 2007 dataset with 0.6 noise rate, run:

```bash
python mcm.py --bs 64 \
              --nc 20 \
              --nepochs 30 \
              --nworkers 1 \
              --dataset voc2007 \
              --seed 42 \
              --root /path/to/your/dataset/ \
              --out /path/to/your/outputs/ \
              --noise_rate 0.6 \
              --lr 1.5e-4 \
              --weight_decay 0.0001
```

Adjust dataset and output paths as needed.

---

## Acknowledgements

This codebase is based on the [HLC](https://github.com/xiaoboxia/HLC/tree/main) code. We thank the authors for their contributions.

---

## Citation

```bibtex
@inproceedings{inproceedings,
  author    = {Gonzalez, Diego and Ibrahim, Shahana},
  title     = {Multi-label Recognition under Noisy Supervision: A Confusion Mixture Modeling Approach},
  booktitle = {ICASSP},
  year      = {2025},
  pages     = {1--5},
  doi       = {10.1109/ICASSP49660.2025.10889155}
}
```
