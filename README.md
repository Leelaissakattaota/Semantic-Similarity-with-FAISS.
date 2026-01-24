# 🔍 Semantic Similarity with FAISS

[![TensorFlow](https://img.shields.io/badge/Powered%20by-TensorFlow%202.0-orange)](https://www.tensorflow.org/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-blue)](https://github.com/facebookresearch/faiss)
[![NLP](https://img.shields.io/badge/NLP-Semantic%20Search-green)](https://tfhub.dev/google/universal-sentence-encoder/4)

This project demonstrates the implementation of a **Semantic Search Engine** using the **Universal Sentence Encoder (USE)** for text vectorization and **FAISS (Facebook AI Similarity Search)** for efficient high-dimensional indexing.

---

## 📖 Project Overview
Traditional search engines rely on keyword matching, which often fails to capture the underlying intent of a user's query. This system implements **Semantic Search**, which understands the context and nuances of human language.

### Key Components:
* **Universal Sentence Encoder (USE)**: Converts preprocessed text into 512-dimensional vectors that represent the semantic essence of sentences.
* **FAISS (Facebook AI Similarity Search)**: An efficient library for rapid similarity search and clustering of dense vectors.
* **20 Newsgroups Dataset**: A collection of ~20,000 documents across 20 topics used to benchmark the search engine.

---

## 🏗️ Technical Workflow
1.  **Preprocessing**: Text is cleaned by removing email headers, email addresses, punctuation, and numbers.
2.  **Vectorization**: Cleaned text is passed through the USE model to create numerical "fingerprints" (vectors).
3.  **Indexing**: Vectors are added to a **FAISS `IndexFlatL2`**, which organizes them in a searchable space using Euclidean distance.
4.  **Querying**: User queries are vectorized in real-time, and FAISS retrieves the nearest neighbors based on semantic similarity.



---

## 🛠️ Technology Stack
* **Language**: `Python`
* **Deep Learning**: `TensorFlow`, `TensorFlow Hub`
* **Vector Search**: `FAISS-cpu`
* **Data Science**: `NumPy`, `scikit-learn`

---

## ⚙️ Setup & Installation
To run this project, install the following dependencies:

```bash
pip install faiss-cpu numpy scikit-learn tensorflow tensorflow-hub
