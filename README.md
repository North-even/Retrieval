# MovieLens Two-Tower Retrieval System

A production-style recommendation retrieval project built on MovieLens 1M Dataset.

This project simulates the **candidate generation (recall) stage** in industrial recommendation systems:

```text
User Request
→ Candidate Retrieval
→ Ranking
→ Final Recommendation
```

This project focuses on:

```text
Retrieval / Matching / ANN Search
```

while my previous Criteo project focuses on:

```text
Ranking / CTR Prediction
```

Together they form a full recommendation pipeline.

---

# 1. Dataset

MovieLens 1M Dataset

Dataset statistics:

| Metric  |     Value |
| ------- | --------: |
| Users   |     6,040 |
| Movies  |     3,952 |
| Ratings | 1,000,209 |

Raw files:

```text
ratings.dat
users.dat
movies.dat
```

---

# 2. Problem Reformulation

MovieLens is an explicit feedback dataset:

```text
rating: 1~5
```

We convert it into implicit recommendation data:

```text
rating >= 4 → positive interaction
rating < 4 → ignore
```

Final processed statistics:

| Metric                |   Value |
| --------------------- | ------: |
| Users                 |   6,038 |
| Effective Items       |   3,533 |
| Positive Interactions | 575,281 |
| Train                 | 563,211 |
| Valid                 |   6,035 |
| Test                  |   6,035 |

---

# 3. Data Split Strategy

Instead of random split:

```text
For each user:
last interaction → test
second last interaction → valid
remaining → train
```

This better simulates real-world recommendation:

```text
predict future preference using historical behavior
```

---

# 4. Project Structure

```text
Retrieval/
├── data/
│   ├── ratings.dat
│   ├── movies.dat
│   ├── users.dat
│   └── processed/
│       ├── train.csv
│       ├── valid.csv
│       ├── test.csv
│       ├── movie_genres.json
│       ├── idx2movie.json
│
├── src/
│   ├── data/
│   │   ├── preprocess_movielens.py
│   │   ├── build_item_features.py
│   │   ├── dataset.py
│   │   └── tower_dataset.py
│   │
│   ├── models/
│   │   ├── mf.py
│   │   └── two_tower.py
│   │
│   ├── train_mf.py
│   ├── evaluate_mf.py
│   ├── train_two_tower.py
│   ├── evaluate_two_tower.py
│   └── faiss_retrieval.py
│
├── outputs/
└── README.md
```

---

# 5. Baseline 1: Matrix Factorization

## Architecture

```text
user_id embedding
item_id embedding
dot product
```

[
score(u,i)=u^Tv
]

---

## Training Objective

BPR Loss:

[
-\log \sigma(score(u,pos)-score(u,neg))
]

---

## Negative Sampling

For each positive interaction:

```text
(user, positive_item)
```

randomly sample:

```text
negative_item
```

that user has never interacted with.

---

## Training Config

| Config        | Value |
| ------------- | ----: |
| Embedding Dim |    64 |
| Batch Size    |  1024 |
| Learning Rate |  1e-3 |
| Epochs        |    10 |

---

## Training Result

| Epoch | Train Loss |
| ----- | ---------: |
| 1     |      0.495 |
| 10    |      0.184 |

---

## Retrieval Evaluation

| Metric    |  Value |
| --------- | -----: |
| Recall@10 | 0.0613 |
| Recall@20 | 0.1044 |
| NDCG@10   | 0.0292 |
| NDCG@20   | 0.0400 |

---

# 6. Baseline 2: Two-Tower Retrieval

To improve retrieval quality, we introduce richer user/item representations.

---

## User Tower

```text
user_id embedding
+
historical watched movie embeddings
```

History aggregation:

```python
history_embedding.mean(dim=1)
```

---

## Item Tower

```text
movie_id embedding
+
genre embedding
```

Movie genres extracted from:

```text
movies.dat
```

Example:

```text
Toy Story → Animation | Children | Comedy
```

---

## Final Matching Score

[
score(u,i)=u_{tower}^T i_{tower}
]

---

## Training Objective

Still uses:

```text
BPR Loss
```

for fair comparison with MF.

---

## Training Result

| Epoch | Train Loss |
| ----- | ---------: |
| 1     |      0.381 |
| 10    |      0.174 |

---

# 7. Two-Tower Evaluation

| Metric    |     MF |  Two-Tower |
| --------- | -----: | ---------: |
| Recall@10 | 0.0613 | **0.0722** |
| Recall@20 | 0.1044 | **0.1201** |
| NDCG@10   | 0.0292 | **0.0354** |
| NDCG@20   | 0.0400 | **0.0473** |

---

# 8. ANN Retrieval with FAISS

Brute-force retrieval:

```text
user vector × all item vectors
```

works for 3k items but is unrealistic in production.

We simulate industrial ANN retrieval using:

FAISS

Pipeline:

```text
item embeddings
→ build FAISS index
→ user embedding query
→ TopK retrieval
```

---

## FAISS Result

| Metric    |  Value |
| --------- | -----: |
| Recall@10 | 0.0721 |
| Recall@20 | 0.1196 |
| NDCG@10   | 0.0353 |
| NDCG@20   | 0.0472 |

Performance is nearly identical to brute-force retrieval while enabling scalable ANN search.

---

# 9. Industrial Mapping

This project simulates real recommendation retrieval systems:

```text
User Request
→ Two-Tower Retrieval
→ FAISS ANN Candidate Generation
→ Ranking Model (DeepFM in previous project)
→ Final Recommendation
```

---

# 10. Key Engineering Challenges Solved

### Time-aware split

Avoided random leakage.

---

### Dynamic negative sampling

Improved retrieval training realism.

---

### User behavior sequence modeling

Used history pooling.

---

### Item side feature engineering

Integrated genre metadata.

---

### ANN deployment simulation

Built scalable retrieval pipeline with FAISS.

---

# 11. Future Improvements

* Hard negative sampling
* In-batch negatives
* Attention-based user modeling
* Transformer sequence modeling
* Multi-interest retrieval
* Online A/B testing simulation
* Cold-start item recommendation

---

# 12. Tech Stack

* Python
* PyTorch
* Pandas
* NumPy
* FAISS
* MovieLens 1M

---

# 13. Resume Highlight

Built an end-to-end recommendation retrieval system based on MovieLens-1M, reformulating explicit ratings into implicit feedback and implementing Matrix Factorization, Two-Tower retrieval, and FAISS ANN search. Improved Recall@20 from **10.44% → 12.01%**, integrated user sequential behavior and item genre features, and simulated industrial-scale candidate generation pipelines used in large-scale recommendation systems.
