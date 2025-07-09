# TASA: Transmitted and Aggregated Self-Attention (My Implementation)

This repository contains **my PyTorch implementation of TASA (Transmitted and Aggregated Self-Attention)**, as proposed in the paper:

> 📄 **[TASA: Transmitted and Aggregated Self-Attention for Speech Recognition](https://www.isca-archive.org/interspeech_2024/zhang24q_interspeech.pdf)**  
> *Zhang, Han, et al. Interspeech 2024.*

---

## 🔍 Overview

TASA enhances the Transformer architecture by introducing two new modules:

- **R-TASA (Residual TASA)**: Accumulates and aggregates self-attention logits from previous layers.
- **D-TASA (Dense TASA)**: Transmits and concatenates logits from all previous layers for richer attention aggregation.

This improves context modeling and allows deeper interactions between layers in the encoder.

---

## 🚀 Features

- Full implementation of **TASA encoder** (both R-TASA and D-TASA variants).
- Compatible with **CTC-based**, **seq2seq (attention-based)**, or **joint CTC + CE** training.
- SpecAugment and speed perturbation support.
- Encoder-decoder architecture based on ESPnet-style configuration.
- Easily switch between AISHELL-1 and LibriSpeech training setups.

---

## 🛠️ Installation

```bash
git clone https://github.com/itsmekhoathekid/TASA.git
cd TASA
pip install -r requirements.txt
