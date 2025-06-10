# 🧠 Introduction to Machine Learning 

![License](https://img.shields.io/badge/License-MIT-blue.svg) ![Build](https://img.shields.io/badge/build-passing-brightgreen) ![Made With ♥](https://img.shields.io/badge/Made%20with-♥-ff69b4)

> **A deep, practical, and modern guide (v3.0 – June 10 2025)**  
> Curated from classic textbooks, top-cited papers, production war stories, and community best practices.  
> **Mission:** Empower readers to design, train, and ship state-of-the-art ML systems responsibly.

---

## 📖 How to Navigate This Guide
1. **Scan the TL;DR** for a holistic map.  
2. **Open corresponding notebooks** in `notebooks/` while reading.  
3. **Follow reference links** for deep dives.  
4. **Use the checklists** at the end of each chapter when working on real projects.

> **Heads-up:** Many code snippets rely on Python ≥3.11 and CUDA-capable GPUs. CPU fallbacks are included but slower.

---

## 🗺️ TL;DR (Click to expand)
<details>
<summary><strong>Concept Roadmap & Ecosystem 2025</strong></summary>

| Step | Theme | Representative Tools / Papers | Key Takeaway |
|------|-------|------------------------------|--------------|
| 1️⃣ | **Math & Stats Primer** | *The Elements of Statistical Learning* | ML ≈ applied statistics + optimisation |
| 2️⃣ | **Supervised Learning** | scikit-learn 1.7, XGBoost 2.0 | 80 % problems solved w/ tabular models |
| 3️⃣ | **Unsupervised Learning** | UMAP [4], HDBSCAN | Pattern discovery + compression |
| 4️⃣ | **Representation Learning** | AutoEncoders, SimCLR | Self-sup beats hand-crafted FE |
| 5️⃣ | **Neural Networks** | Transformer [5], Mamba (2024) | Attention still king but efficiency matters |
| 6️⃣ | **Computer Vision** | ResNet, YOLOv9 [7], SAM [8] | Vision solved? Not for long-tail edge cases |
| 7️⃣ | **NLP / LLMs** | GPT-4o, Mixtral-8x22B | LLMs → multimodal + tool-aware |
| 8️⃣ | **Graph ML** | GraphSAGE, GNN Explainer | Entities + relations > isolated samples |
| 9️⃣ | **Reinforcement Learning** | PPO, MuZero | Optimal sequential decisions |
| 🔟 | **Generative (Diffusion)** | Stable Diffusion 3, SVD | SOTA image + video generation |
| 1️⃣1️⃣ | **AutoML** | Auto-sklearn 2, AutoKeras 1.1 | Automation accelerates iteration |
| 1️⃣2️⃣ | **MLOps** | MLflow 3, Evidently AI | Repro, deploy, monitor |
| 1️⃣3️⃣ | **Responsible AI** | EU AI Act [16], fairness metrics | Safety & compliance baked-in |

</details>

---

## 📑 Extended Table of Contents
1. [Math, Stats & Linear Algebra Refresher](#1-math-stats--linear-algebra-refresher)  
2. [Supervised Learning](#2-supervised-learning)  
3. [Unsupervised & Self-Supervised Learning](#3-unsupervised--self-supervised-learning)  
4. [Representation Learning](#4-representation-learning)  
5. [Neural Networks](#5-neural-networks)  
6. [Computer Vision](#6-computer-vision)  
7. [Natural Language Processing](#7-natural-language-processing)  
8. [Graph Machine Learning](#8-graph-machine-learning)  
9. [Reinforcement Learning](#9-reinforcement-learning)  
10. [Generative Models (Diffusion)](#10-generative-models-diffusion)  
11. [AutoML & Meta-Learning](#11-automl--meta-learning)  
12. [Model Evaluation & HPO](#12-model-evaluation--hpo)  
13. [MLOps & Production](#13-mlops--production)  
14. [Ethics, Privacy & Responsible AI](#14-ethics-privacy--responsible-ai)  
15. [Installation](#15-installation)  
16. [Usage](#16-usage)  
17. [Contribution Guide](#17-contribution-guide)  
18. [License](#18-license)  
19. [References & Further Reading](#19-references--further-reading)

---

## 1 · Math, Stats & Linear Algebra Refresher
![Math & Stats](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Vector_Addition.svg/2560px-Vector_Addition.svg.png)

* **Probability:** Bayes’ theorem, distributions, KL-divergence.  
* **Linear Algebra:** vectors, matrices, eigen-decomposition.  
* **Optimisation:** gradient descent, convexity, Lagrange multipliers.  
* **Information Theory:** entropy, mutual information.

<details>
<summary>📚 Key Resources</summary>

* *Pattern Recognition & Machine Learning* – Bishop [2]  
* “Convex Optimisation” – Boyd & Vandenberghe (free PDF)  
* Stanford CS229 lecture notes (2024)  

</details>

---

## 2 · Supervised Learning
Standard workflow now includes **data-centric AI** practices:

1. **Label quality audits** using `cleanlab`.  
2. **Feature stores** (Feast) to unify offline/online logic.  
3. **Model training** – start with tree ensembles → escalate to DL if needed.

### 2.1 Modern Best-Practice Checklist
- [ ] Stratified split or time-series split  
- [ ] Leakage tests (target leakage, train/serving skew)  
- [ ] Baseline model + “null” model  
- [ ] Hyperparameter search with cross-validation  
- [ ] Calibration (Platt scaling / isotonic)  
- [ ] Explainability (SHAP, feature importance)

```python
import cleanlab, shap, xgboost
# Example pipeline omitted for brevity – see notebooks/supervised.ipynb
3 · Unsupervised & Self-Supervised Learning
Beyond traditional clustering, self-supervised objectives (contrastive, masked prediction) unlock signal from unlabeled data.

Contrastive Learning: SimCLR, BYOL, MoCo-v3.

Masked Modelling: MAE for vision, BERT for text.

Anomaly Detection: Isolation Forest, AutoEncoder reconstruction error.

Insight: In many domains, self-sup pre-training + small labelled finetune outperforms fully supervised models [17].

4 · Representation Learning
AutoEncoders – compress & reconstruct.

Latent Variable Models: VAEs, normalising flows.

Metric Learning: Triplet loss, ArcFace.

5 · Neural Networks
5.1 Recent Innovations (2023-2025)
Theme	Paper / Tech	Summary
Efficiency	Flash-Attention-2	Memory-optimal attention kernels
Long-Context	Mamba, RWKV-6	Linear RNN hybrids, 32k tokens
Sparsity	MoE (Mixtral-8x22B)	45 % FLOPs vs. dense LLM
Quantisation	GPTQ, AWQ	8-bit / 4-bit inference on edge

6 · Computer Vision
Add foundation models like DINOv2 (self-sup vision) and SAM (universal segmentation).


Example fine-tuning script under src/cv/:

bash
Copy
Edit
python src/cv/finetune_dinov2.py --dataset flowers102 --epochs 30
7 · Natural Language Processing
Retrieval-Augmented Generation (RAG): LangChain, LlamaIndex.

Evaluation: BLEU obsolete → COMET-Kiwi 2.0, GPT-Score.

8 · Graph Machine Learning
Graph neural networks (GNNs) generalise convolutions to graph-structured data.

python
Copy
Edit
import torch_geometric as tg
from torch_geometric.nn import GraphSAGE
Applications: social networks, knowledge graphs, recommender systems.

Explainability: GNNExplainer, GraphSVX.

9 · Reinforcement Learning
Algorithms: PPO (stable), SAC (continuous), MuZero (model-based).

Frameworks: RLlib 3, CleanRL 2.

Simulators: OpenAI Gymnasium, DeepMind DM-Control.

Trend: Offline RL + diffusion policies match online data efficiency [18].

10 · Generative Models (Diffusion)
Text-to-3D: DreamFusion, Gaussian Splatting 2025.

Video diffusion: SVD-X4 generates 4K 30 fps clips.

11 · AutoML & Meta-Learning
AutoML democratises ML by automating pipeline design.

Search spaces: model, preprocessing, HPO.

Zero-shot AutoML: match dataset meta-features to prior runs.

Meta-learning: MAML, Reptile for quick adaptation.

12 · Model Evaluation & HPO
See src/hpo/ for Bayesian Opt via optuna, ray.tune. Key metrics list extended in Appendix A.

13 · MLOps & Production
Add LLMOps patterns (prompt store, caching, guardrails). Provide Dockerfile + Makefile for infra-as-code.

14 · Ethics, Privacy & Responsible AI
Fairness metrics: demographic parity, equalised odds.

Privacy: differential privacy (Opacus), federated learning (Flower 2).

Red-teaming: automated adversarial testing harness.

15 · Installation
Supports pip, conda, and Docker. Use:

bash
Copy
Edit
make dev   # sets up venv + installs deps
make gpu   # installs GPU extras
16 · Usage
Jupyter Notebooks – interactive demos.

CLI – python -m ml_intro.train --help.

REST API – FastAPI server under serving/.

17 · Contribution Guide
Standard GitHub flow + pre-commit hooks:

bash
Copy
Edit
pre-commit install
18 · License
Distributed under the MIT License – see LICENSE.

19 · References & Further Reading
I. Goodfellow, Y. Bengio, A. Courville. Deep Learning. MIT Press, 2016.

C. Bishop. Pattern Recognition & ML. Springer, 2006.

scikit-learn documentation (v1.7).

L. McInnes et al. “UMAP.” 2018.

A. Vaswani et al. “Attention Is All You Need.” 2017.

M. Tan & Q. Le. “EfficientNetV2.” 2021.

Ultralytics. “YOLOv9.” 2024.

Meta AI. “Segment Anything.” 2023.

OpenAI. “GPT-4o Technical Report.” 2025.

W. Li et al. “Diffusion Models Survey.” 2024.

A. Radford et al. “CLIP.” 2021.

DataCamp. “MLOps Tools 2025.” 2024.

Z. Liu et al. “Flash-Attention-2.” 2023.

UNESCO. Ethics of AI. 2024.

EU Parliament. AI Act. 2024.

B. Zoph et al. “AutoML Techniques Survey.” 2025.

J. Chen et al. “Self-Supervised Learning – A Systematic Review.” 2025.

A. Kumar et al. “Offline RL with Diffusion Models.” 2024.
