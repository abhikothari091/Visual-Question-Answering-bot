# Visual Question Answering Bot

This project is a multi-modal Visual Question Answering (VQA) system designed to interpret images and answer questions about them. By combining computer vision and natural language processing, the model analyzes visual inputs and textual queries to provide accurate responses. The project leverages state-of-the-art techniques and frameworks, including ResNet-50 for image feature extraction and BERT for text embeddings.

---

## Table of Contents
1. [Dataset Information](#dataset-information)
2. [Model Architecture](#model-architecture)
3. [Use Cases](#use-cases)
4. [Implementation](#implementation)
5. [Performance](#performance)
6. [Challenges and Future Work](#challenges-and-future-work)
7. [Acknowledgments](#acknowledgments)

---

## Dataset Information

We utilized the **VQA v2 dataset**, a comprehensive dataset for training and evaluating VQA models.

- **Preprocessing**:
  - Data is batched according to image-question-answer mappings.
  - Top 1,000 most frequently occurring answer classes were identified and used as output labels. This ensures a focus on high-frequency classes while maintaining computational efficiency.

---

## Model Architecture

Our model integrates advanced architectures for both image and text processing:

### 1. **Image Tokenization (ResNet-50)**
- Pre-trained **ResNet-50** was fine-tuned by modifying its last layer.
- Image embeddings are mapped to the hidden dimensional size using the attention fusion layer.

### 2. **Text Tokenization (BERT)**
- Tokenizes input questions and generates embeddings.
- Extracted features are passed through a transformer-based network for contextual representation.

### 3. **Attention Fusion Layer**
- Combines image and text embeddings for joint reasoning.
- Approaches tried:
  - **Stacking**: Simple concatenation of features.
  - **Cross Attention**: Image and text features interact directly to enhance joint understanding.
  - **Skip Connection**: Added residual connections to avoid information bottlenecks, which provided the best validation accuracy.

## Use Cases

Potential applications of this project include:
1. **Assistive Technology**: Helping visually impaired individuals by answering questions about images.
2. **Educational Tools**: Supporting interactive learning by enabling visual context-based question answering.
3. **Content Moderation**: Automating responses to image-based queries in social media platforms.
4. **Customer Support**: Enabling visual support in retail or e-commerce scenarios.

---

## Performance

The model achieved **61% accuracy** on the validation set. This was computed using cross-entropy loss and evaluated on a subset of the VQA v2 dataset.

**Highlights**:
- Best performance was achieved using skip connections in the attention fusion layer.
- Strong results in answering yes/no questions and questions about colors.

---

## Challenges and Future Work

### Challenges:
- **Bias**: The model occasionally shows bias toward high-frequency answers, especially for color-based questions.
- **Complexity**: Struggles to answer questions requiring intricate reasoning beyond the scope of the top 1,000 answer classes.

### Future Work:
1. Expand the output classes to cover more diverse and complex questions.
2. Implement additional data augmentation techniques to improve generalization.
3. Explore advanced attention mechanisms like dynamic attention weighting.

---

## Acknowledgments

This project was a part of the **Supervised Machine Learning (DS 5220)** course at **Northeastern University**.
We extend our gratitude to:
- **Professor Ryan Zhang** for his invaluable support and guidance.
- **Teaching Assistants** for their continuous feedback.
- **Northeastern University** for providing the necessary resources.

---

Feel free to contribute to this project or raise issues for further discussion!
```
