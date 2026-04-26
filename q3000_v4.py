# -*- coding: utf-8 -*-
"""
Q3000 v4 - Quaternion Embedding Sentiment Classifier
Author: [Your Name]

Uses quaternion embeddings with Hamilton product interactions and
attention pooling to classify IMDb movie review sentiment.
Motivation: Geometric representations in quaternion space can encode
richer relational structure between token embeddings than standard
flat vector embeddings.

v2 changes: Added Dropout(0.3) after attention pooling to reduce overfitting.
v3 changes: Increased MAX_LEN from 100 → 300 so the model reads more of each
            review. IMDb reviews are often long and sentiment is expressed late —
            truncating at 100 tokens was leaving most of the text unread.
v4 changes: Added EarlyStopping to stop training at peak validation accuracy.
            Added standard Embedding baseline to prove quaternion embeddings
            outperform a conventional approach.
"""

# ─────────────────────────────────────────────
# STEP 1: Install & Import
# ─────────────────────────────────────────────
!pip install datasets scikit-learn --quiet

from datasets import load_dataset
import re
from collections import Counter
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Softmax, Multiply, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ─────────────────────────────────────────────
# STEP 2: Load IMDb Dataset (Train + Test)
# ─────────────────────────────────────────────
print("Loading IMDb dataset...")
train_dataset = load_dataset("imdb", split="train")  # 25K
test_dataset  = load_dataset("imdb", split="test")   # 25K held-out

train_texts  = [s["text"]  for s in train_dataset]
train_labels = [s["label"] for s in train_dataset]
test_texts   = [s["text"]  for s in test_dataset]
test_labels  = [s["label"] for s in test_dataset]

# ─────────────────────────────────────────────
# STEP 3: Tokenize & Build Vocabulary
# ─────────────────────────────────────────────
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

print("Tokenizing...")
tokenized_train = [tokenize(t) for t in train_texts]
tokenized_test  = [tokenize(t) for t in test_texts]

# Build vocab from top 10,000 words in TRAINING set only
word_counts = Counter(word for sentence in tokenized_train for word in sentence)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common(10000))}
# idx+1 so that 0 is reserved as the padding token

MAX_LEN    = 300
VOCAB_SIZE = len(vocab) + 1  # +1 for padding index 0

def encode(tokenized_texts, vocab, max_len):
    ids = [[vocab.get(word, 0) for word in tokens] for tokens in tokenized_texts]
    return pad_sequences(ids, maxlen=max_len, padding='post', truncating='post')

X_train = encode(tokenized_train, vocab, MAX_LEN)
X_test  = encode(tokenized_test,  vocab, MAX_LEN)
y_train = np.array(train_labels).reshape(-1, 1)
y_test  = np.array(test_labels).reshape(-1, 1)

print(f"✅ Data ready | X_train: {X_train.shape} | X_test: {X_test.shape}")

# ─────────────────────────────────────────────
# STEP 4: Quaternion Embedding Layer
#
# Each token is represented as a quaternion q = r + ii + jj + kk
# The Hamilton product between two quaternions q1 and q2 is:
#   r  = r1*r2 - i1*i2 - j1*j2 - k1*k2
#   i  = r1*i2 + i1*r2 + j1*k2 - k1*j2
#   j  = r1*j2 - i1*k2 + j1*r2 + k1*i2
#   k  = r1*k2 + i1*j2 - j1*i2 + k1*r2
#
# This encodes non-commutative geometric interactions between
# components — unlike standard embeddings which are just dot products.
# ─────────────────────────────────────────────
class QuaternionEmbedding(Layer):
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        init = tf.keras.initializers.GlorotUniform()
        self.r = self.add_weight(shape=(self.vocab_size, self.embedding_dim), initializer=init, name="r")
        self.i = self.add_weight(shape=(self.vocab_size, self.embedding_dim), initializer=init, name="i")
        self.j = self.add_weight(shape=(self.vocab_size, self.embedding_dim), initializer=init, name="j")
        self.k = self.add_weight(shape=(self.vocab_size, self.embedding_dim), initializer=init, name="k")

    def call(self, inputs):
        r = tf.nn.embedding_lookup(self.r, inputs)
        i = tf.nn.embedding_lookup(self.i, inputs)
        j = tf.nn.embedding_lookup(self.j, inputs)
        k = tf.nn.embedding_lookup(self.k, inputs)

        # Self-Hamilton product: q * q
        hr = r*r - i*i - j*j - k*k
        hi = r*i + i*r + j*k - k*j
        hj = r*j - i*k + j*r + k*i
        hk = r*k + i*j - j*i + k*r

        return tf.concat([hr, hi, hj, hk], axis=-1)  # (batch, seq, 4*dim)

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size, "embedding_dim": self.embedding_dim})
        return config


# ─────────────────────────────────────────────
# STEP 5: Attention Pooling Layer
# ─────────────────────────────────────────────
class AttentionPooling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(1)

    def call(self, inputs):
        scores   = self.dense(inputs)
        weights  = Softmax(axis=1)(scores)
        weighted = Multiply()([inputs, weights])
        return tf.reduce_sum(weighted, axis=1)

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────
# STEP 6: EarlyStopping Callback
#
# Monitors validation accuracy and stops training when it stops
# improving. restore_best_weights=True means we keep the weights
# from the best epoch, not the last one.
# ─────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=2,                  # stop after 2 epochs with no improvement
    restore_best_weights=True,   # revert to best epoch weights
    verbose=1
)

EMBEDDING_DIM = 16  # quaternion expands this to 4*16 = 64 effective dims


# ─────────────────────────────────────────────
# STEP 7: Train Quaternion Model
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  MODEL A: Quaternion Embedding + Attention Pooling")
print("="*55)

input_layer = Input(shape=(MAX_LEN,), dtype=tf.int32)
x = QuaternionEmbedding(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)(input_layer)
x = AttentionPooling()(x)
x = Dropout(0.3)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(1, activation="sigmoid")(x)

quat_model = Model(inputs=input_layer, outputs=output)
quat_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
quat_model.summary()

print("\nTraining Quaternion model...")
quat_model.fit(
    X_train, y_train,
    epochs=10,                   # allow up to 10, EarlyStopping will cut it short
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

print("\n📊 Evaluating Quaternion model...")
quat_pred_prob = quat_model.predict(X_test, batch_size=128)
quat_pred      = (quat_pred_prob > 0.5).astype(int).flatten()
quat_acc       = accuracy_score(y_test.flatten(), quat_pred)
quat_f1        = f1_score(y_test.flatten(), quat_pred)
print(f"✅ Quaternion Test Accuracy : {quat_acc:.4f}")
print(f"✅ Quaternion Test F1 Score : {quat_f1:.4f}")


# ─────────────────────────────────────────────
# STEP 8: Train Standard Embedding Baseline
#
# Same architecture (attention pooling, dropout, dense layers)
# but uses a conventional Keras Embedding layer instead of
# quaternion embeddings. Fair comparison — only the embedding
# layer differs. Embedding dim = 64 to match quaternion's
# effective 4*16 = 64 output dimensions.
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  MODEL B: Standard Embedding Baseline (control)")
print("="*55)

early_stop_baseline = EarlyStopping(
    monitor='val_accuracy',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

input_layer_b = Input(shape=(MAX_LEN,), dtype=tf.int32)
x_b = Embedding(input_dim=VOCAB_SIZE, output_dim=64)(input_layer_b)
x_b = AttentionPooling()(x_b)
x_b = Dropout(0.3)(x_b)
x_b = Dense(32, activation="relu")(x_b)
x_b = Dropout(0.2)(x_b)
output_b = Dense(1, activation="sigmoid")(x_b)

baseline_model = Model(inputs=input_layer_b, outputs=output_b)
baseline_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
baseline_model.summary()

print("\nTraining Baseline model...")
baseline_model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop_baseline],
    verbose=1
)

print("\n📊 Evaluating Baseline model...")
base_pred_prob = baseline_model.predict(X_test, batch_size=128)
base_pred      = (base_pred_prob > 0.5).astype(int).flatten()
base_acc       = accuracy_score(y_test.flatten(), base_pred)
base_f1        = f1_score(y_test.flatten(), base_pred)
print(f"✅ Baseline Test Accuracy : {base_acc:.4f}")
print(f"✅ Baseline Test F1 Score : {base_f1:.4f}")


# ─────────────────────────────────────────────
# STEP 9: Results Comparison
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  FINAL RESULTS COMPARISON")
print("="*55)
print(f"{'Model':<35} {'Accuracy':>10} {'F1':>10}")
print("-"*55)
print(f"{'Quaternion Embedding (Q3000)':<35} {quat_acc:>10.4f} {quat_f1:>10.4f}")
print(f"{'Standard Embedding (Baseline)':<35} {base_acc:>10.4f} {base_f1:>10.4f}")
print("-"*55)
diff = quat_acc - base_acc
winner = "Quaternion ✅" if diff > 0 else "Baseline"
print(f"\nWinner: {winner}  (Δ accuracy: {diff:+.4f})")


# ─────────────────────────────────────────────
# STEP 10: Custom Inference (Quaternion Model)
# ─────────────────────────────────────────────
def predict_sentiment(text, model, vocab, max_len=300):
    tokens = [vocab.get(word, 0) for word in tokenize(text)]
    padded = pad_sequences([tokens], maxlen=max_len, padding='post', truncating='post')
    prob   = model.predict(padded, verbose=0)[0][0]
    label  = "Positive ✅" if prob > 0.5 else "Negative ❌"
    print(f"\n\"{text[:80]}...\"" if len(text) > 80 else f"\n\"{text}\"")
    print(f"  → {label}  (confidence: {prob:.4f})")
    return prob

print("\n─── Custom Inference Tests (Quaternion Model) ───")
predict_sentiment("Absolutely loved it. Brilliant acting, perfect pacing, and I'd watch it again tomorrow.", quat_model, vocab)
predict_sentiment("Terrible from start to finish. Wooden acting, awful script, and I regret watching it.", quat_model, vocab)
predict_sentiment("It took a while to get going, but by the end I found myself smiling. Surprisingly heartfelt.", quat_model, vocab)
predict_sentiment("The film was well-made and the performances were decent. It didn't really leave a strong impression, but it was fine.", quat_model, vocab)
predict_sentiment("I really wanted to like it. The trailer looked great, but the final product just didn't connect with me.", quat_model, vocab)
