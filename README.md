# Q3000
Quaternion embedding model for sentiment analysis using Hamilton product interactions
Q3000 — Quaternion Embedding Sentiment Classifier
> *"Binary seems archaic. We should be programming in geometry."*
A custom sentiment analysis model built from scratch using quaternion embeddings and Hamilton product interactions. Trained and evaluated on the IMDb movie review dataset.
---
Motivation
Most neural networks represent words as flat vectors — points in a high-dimensional space with no inherent geometric structure. I wanted to explore whether representing words as quaternions (4-component mathematical objects that encode rotation and orientation in 3D space) could capture richer relational structure between tokens.
Quaternions are used in physics, robotics, and 3D graphics to represent rotation in ways that flat vectors cannot. The hypothesis: if language has geometric structure, quaternion space might be a more natural home for it than standard Euclidean embeddings.
This project was built to test that hypothesis — and the results suggest it holds under controlled conditions.
---
Architecture
![Q3000 Architecture](architecture.png)
```
Input (token IDs)
        ↓
QuaternionEmbedding     ← custom layer: 4 weight matrices (r, i, j, k)
        ↓                  Hamilton product applied: q * q
        ↓                  output: (batch, seq_len, 64)
AttentionPooling        ← learns which tokens matter most for sentiment
        ↓
Dropout(0.3)
        ↓
Dense(32, relu)
        ↓
Dropout(0.2)
        ↓
Dense(1, sigmoid)       ← binary output: positive / negative
```
Quaternion Embedding Layer
Each token is represented as a quaternion `q = r + i·i + j·j + k·k` with four learned weight components. The Hamilton product (q × q) is applied to create geometric interactions between components:
```
hr = r·r - i·i - j·j - k·k
hi = r·i + i·r + j·k - k·j
hj = r·j - i·k + j·r + k·i
hk = r·k + i·j - j·i + k·r
```
This is fundamentally different from standard embeddings, which simply look up a vector with no cross-component interaction. The Hamilton product is non-commutative — meaning order matters — which may help capture directional sentiment relationships.
Attention Pooling
Rather than mean or max pooling, the model learns a weighted sum over the sequence — assigning higher weight to tokens that are more sentiment-relevant. This is especially important with `MAX_LEN=300`, where the model must identify which parts of a long review carry the most signal.
---
Results
Model	Test Accuracy	F1 Score
Q3000 (Quaternion Embedding)	87.54%	0.8755
Baseline (Standard Embedding)	86.66%	0.8663
Δ	+0.88%	+0.0092
Both models use identical architectures (attention pooling, dropout, dense layers) and identical parameter counts (642,242). The only difference is the embedding layer.
The improvement is modest, but consistent across identical conditions, suggesting the embedding structure contributes meaningful signal. EarlyStopping (`patience=2, restore_best_weights=True`) was used to control overfitting and ensure the comparison reflects generalization performance.
Version History
Version	Change	Test Accuracy
v1	Initial build, attention pooling	80.00%
v2	+ Dropout(0.3, 0.2)	80.27%
v3	MAX_LEN: 100 → 300	86.30%
v4	+ EarlyStopping + Baseline comparison	87.54%
---
Key Findings
Geometric embeddings slightly outperform flat vector embeddings under identical conditions: Quaternion embeddings outperformed a standard embedding baseline by 0.88% with no additional parameters or architectural complexity.
Context length matters most: The single biggest jump (+6%) came from increasing MAX_LEN from 100 to 300 — the model was simply not reading enough of each review.
Attention earns its keep: With 300 tokens to process, the attention layer can identify sentiment-rich sentences within long reviews rather than averaging everything equally.
---
Setup & Usage
```bash
pip install datasets scikit-learn tensorflow
```
Run in Google Colab or any environment with TensorFlow 2.x:
```bash
python q3000_v4.py
```
The script will:
Download the IMDb dataset automatically
Train the quaternion model with early stopping
Train the standard embedding baseline
Print a side-by-side comparison
Run inference on five sample reviews
Custom Inference
```python
predict_sentiment("Your review text here.", quat_model, vocab)
# → Positive ✅  (confidence: 0.9821)
```
---
Dataset
IMDb Large Movie Review Dataset
25,000 training reviews / 25,000 test reviews
Binary labels: positive (1) or negative (0)
Loaded via HuggingFace `datasets` library
---
Tech Stack
Python 3.12
TensorFlow / Keras
HuggingFace Datasets
scikit-learn (evaluation metrics)
Google Colab (training environment)
---
What's Next
Token-to-token Hamilton products (cross-token geometric interaction)
Mask-aware attention pooling (ignore padding tokens)
Comparison against pretrained embeddings (GloVe, Word2Vec)
Extension to multi-class sentiment (1–5 star rating prediction)
---
About
Built as a first coding project during Year 1 of an Associate of Science in AI. Started with an intuition about geometric representations in language — before knowing how to code — and worked toward proving it experimentally.
If RoPE (Rotary Position Embeddings) can encode position as rotation in modern LLMs, maybe sentiment has geometric structure too.
