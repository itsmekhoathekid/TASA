# R-TASA × SpeechTransformer (PyTorch)

PyTorch implementation of **Residual Transmitted & Aggregated Self-Attention (R-TASA)** integrated into **SpeechTransformer** for ASR.

> 📄 Zhang, Han, et al. “TASA: Transmitted and Aggregated Self-Attention for Speech Recognition.” *INTERSPEECH 2024*. [[paper]](https://www.isca-archive.org/interspeech_2024/zhang24q_interspeech.pdf)  
> 📄 Dong, Linhao, et al. “Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition.” *ICASSP 2018*. [[paper]](https://ieeexplore.ieee.org/document/8462506)

---

## 🛠️ Install

```bash
git clone https://github.com/itsmekhoathekid/TASA.git
cd TASA
pip install -r requirements.txt
```

---

## ⚙️ Usage

Enable R-TASA in config:

```yaml
model:
  attention_type: "r_tasa" 
```

Train example:

```bash
python train.py --config config/r_tasa_local.yaml
```

---
