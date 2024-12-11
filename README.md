# Visual Question Answering Bot

**Team members: Abhishek Kothari, Vignesh Ramaswamy Balasundaram, Yaksh Shah.**
This project is a multi-modal **Visual Question Answering (VQA)** system that interprets images and answers questions about them. By leveraging computer vision and natural language processing techniques, the model combines visual and textual understanding to provide meaningful responses. The core architecture integrates ResNet-50 for image feature extraction, BERT for text embeddings, and an attention fusion layer to merge both modalities.

---

## Table of Contents
1. [Dataset Information](#dataset-information)
2. [Model Architecture](#model-architecture)
3. [Use Cases](#use-cases)
4. [Performance Metrics](#performance-metrics)
5. [Limitations of the Model](#limitations-of-the-model)
6. [Challenges and Future Directions](#challenges-and-future-directions)
7. [Lessons and Insights](#lessons-and-insights)
8. [Acknowledgments](#acknowledgments)

---

## Dataset Information

We use the **VQA v2 dataset**, a widely recognized benchmark for Visual Question Answering tasks. The dataset includes images from MS COCO and questions with corresponding answers, curated to evaluate a model's reasoning capabilities.The dataset used in this project can be downloaded from this [link](https://visualqa.org/vqa_v1_download.html)

### Preprocessing
- **Data Loader**:
  - Images are grouped into batches based on their associated questions and answers.
  - Custom logic extracts the top 1,000 most frequently occurring answers, which form the model's output classes.
  - Rare answers were excluded to streamline training and improve focus on high-frequency classes.
- **Tokenization**:
  - Each question is tokenized and transformed into embeddings using BERT.
  - Images are converted into feature maps using ResNet-50.

---

## Model Architecture

### 1. **Image Feature Extraction (ResNet-50)**
- A pre-trained **ResNet-50** model was fine-tuned by modifying its last layer to adapt to this specific task.
- The extracted image features were projected into a fixed-dimensional space matching the hidden dimension of the attention fusion layer.
- This ensures compatibility when combining with text embeddings.

### 2. **Text Embedding Generation (BERT)**
- We use **BERT-Base** to tokenize and generate contextual embeddings for the input questions.
- Each question is mapped to a high-dimensional feature vector representing its semantic meaning.

### 3. **Attention Fusion Layer**
- Combines image and text embeddings for joint reasoning.
- Fusion strategies explored:
  1. **Stacking**: Simple concatenation of image and text features.
  2. **Cross Attention**: Mutual interaction between image and text features to amplify shared information.
  3. **Skip Connections**: Residual connections added to preserve the original features and prevent bottlenecks. This approach yielded the best accuracy.

---

## Use Cases

The VQA Bot has several potential applications:
1. **Assistive Technology**:
   - Helping visually impaired individuals answer questions about their surroundings.
2. **Education**:
   - Enabling interactive, image-based learning experiences.
3. **Content Moderation**:
   - Automating query responses for visual content in social media or e-commerce.
4. **Retail/Customer Support**:
   - Providing answers about products and services based on uploaded images.

---

## Performance Metrics

The model achieved a **validation accuracy of 61%**, evaluated on the top 1,000 answer classes from the VQA v2 dataset.

**Key Observations**:
- High accuracy in answering binary (yes/no) and color-related questions.
- Robust performance in identifying objects and simple attributes in images.
- The skip connection-based attention fusion layer provided the most consistent improvements across various question types.

---

## Limitations of the Model

1. **Restricted Answer Space**:  
   The model is limited to a predefined set of 1,000 possible answers. If the correct answer is not included in this set, the model will predict an unrelated or incorrect answer.  
   **Example:**  
   - **User Question:** What is the color of the t-shirt the person is wearing in the image ?  
   - **True Answer:** Pink  
   - **Predicted Answer:** Black (if "Pink" is not part of the predefined answer set).  

2. **Single-Word Responses**:  
   Currently, the model is designed to provide only one-word answers, which restricts its ability to handle complex or descriptive queries.  

These limitations highlight the need for further development to expand the answer set and support more expressive responses.


---
## Challenges and Future Directions

### Challenges
1. **Class Bias**:
   - The model shows a tendency to over-predict answers corresponding to high-frequency classes.
2. **Complexity of Questions**:
   - Struggles with reasoning-intensive queries that go beyond basic visual or textual comprehension.

### Future Directions
1. **Expand Output Classes**:
   - Incorporate more diverse and complex answer classes to improve generalization.
2. **Dynamic Attention Mechanisms**:
   - Experiment with adaptive attention layers for better feature alignment.
3. **Data Augmentation**:
   - Introduce synthetic data to improve model robustness and performance on rare questions.

---
## Lessons and Insights

1. **Importance of Answer Set Design**:  
   - Defining the answer space is critical for the accuracy of a Visual Question Answering (VQA) model. A limited or incomplete answer set can lead to incorrect predictions even when the model processes inputs correctly.  
   - Future implementations should consider dynamically expanding the answer set or leveraging open vocabulary approaches to enhance flexibility.  

2. **Multimodal Learning Challenges**:  
   - Combining text and image features requires careful alignment and integration. Multihead Attention proved effective, but fine-tuning the interaction between these modalities remains complex and impactful on performance.  

3. **Trade-offs in Model Simplicity**:  
   - Limiting responses to one-word answers simplifies the output space but restricts the model's applicability for nuanced or detailed queries. Supporting multi-word answers could unlock broader use cases but adds complexity to training and evaluation.  

4. **Pretrained Model Advantages**:  
   - Leveraging BERT for text encoding and ResNet-50 for image feature extraction significantly accelerated development while ensuring robust performance for their respective tasks.  

5. **Error Analysis and Improvement Scope**:  
   - Error analysis revealed the need to handle out-of-vocabulary answers and edge cases where the correct answer is unavailable. These insights emphasize the importance of iterative improvements and dynamic testing methods in VQA systems.  

6. **Application Potential**:  
   - This project demonstrated the practical applications of multimodal deep learning in areas such as interactive systems, image-based search, and AI-driven assistance, paving the way for real-world deployment with further refinements.  

By addressing these lessons, the project can evolve into a more flexible and accurate solution, better equipped to meet diverse user needs.

---

## Acknowledgments

This project was completed as part of the **Supervised Machine Learning (DS 5220)** course at **Northeastern University**.  

We extend our heartfelt gratitude to:
- **Professor Ryan Zhang** for his invaluable support and mentorship.
- **Teaching Assistants** for their constant guidance and feedback.
- **Northeastern University** for providing the necessary resources and infrastructure.

---

Feel free to explore, fork, and contribute to this repository. If you encounter any issues, don’t hesitate to raise them in the **Issues** section!

