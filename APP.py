import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet50
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Image Feature Extractor
# -------------------------
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        resnet = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove classification layer

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features.squeeze(-1).squeeze(-1)  # Shape: [batch_size, 2048]

# -------------------------
# Question Encoder
# -------------------------
class QuestionEncoder(nn.Module):
    def __init__(self):
        super(QuestionEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")

    def forward(self, questions):
        tokenized = self.tokenizer(
            questions, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        outputs = self.bert(**tokenized)
        return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embeddings

# -------------------------
# Attention Fusion Module
# -------------------------
class AttentionFusion(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, image_features, text_features):
        img_proj = self.image_proj(image_features)
        txt_proj = self.text_proj(text_features)
        combined_features = torch.stack([img_proj, txt_proj], dim=1)
        attention_output, _ = self.attention(
            combined_features, combined_features, combined_features
        )
        return attention_output.mean(dim=1)  # Aggregate features

# -------------------------
# Answer Predictor
# -------------------------
class AnswerPredictor(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(AnswerPredictor, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# -------------------------
# VQA Model
# -------------------------
class VQAModel(nn.Module):
    def __init__(self, num_classes):
        super(VQAModel, self).__init__()
        self.image_encoder = ImageFeatureExtractor()
        self.question_encoder = QuestionEncoder()
        self.fusion = AttentionFusion(2048, 768, 512)
        self.classifier = AnswerPredictor(512, num_classes)

    def forward(self, images, questions):
        img_features = self.image_encoder(images)
        ques_features = self.question_encoder(questions)
        fused_features = self.fusion(img_features, ques_features)
        output = self.classifier(fused_features)
        return output

# -------------------------
# Load and Reverse Answer Mapping
# -------------------------
def load_answer_mapping(answer_map_path):
    with open(answer_map_path, 'r') as f:
        answer_map = json.load(f)
    # Reverse the mapping to map indices to answers
    reversed_map = {v: k for k, v in answer_map.items()}
    return reversed_map

# -------------------------
# Load model and weights
# -------------------------
num_classes = 1000  # Adjust as per training
model = VQAModel(num_classes=num_classes).to(DEVICE)
weights_path = "C:/Users/DELL/Downloads/VQA_Model_1.pth"
model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
model.eval()

# -------------------------
# Streamlit App
# -------------------------
st.title("🌟 Visual Question Answering WebApp 🌟")
st.write("Upload an image and ask a question to get an AI-powered answer!")

# Sidebar instructions
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Upload an image using the uploader below.
    2. Enter a question about the image in the text box.
    3. Click the **Submit** button to get an AI-predicted answer.
    """)
    st.write("Example questions: *What is the object in the image?*, *How many people are there?*")

# Image Upload
img_file = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Question Input
question = st.text_input("Ask a question about the image:")

# Load reverse answer mapping
answer_map_path = "C:\\Users\\DELL\\Downloads\\top1000_answer_vocab_dict.json"  # Update with the correct file path
answer_mapping = load_answer_mapping(answer_map_path)

# Submit Button
if st.button("Submit"):
    if not img_file or not question:
        st.warning("Please upload an image and enter a question.")
    else:
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_image = transform(image).unsqueeze(0).to(DEVICE)

        # Question preprocessing
        question_text = [question]  # Single question as a list
        with torch.no_grad():
            prediction = model(input_image, question_text)
            predicted_class = torch.argmax(prediction, dim=-1).item()

        # Map prediction to answer using reversed mapping
        predicted_answer = answer_mapping.get(predicted_class, "Unknown Answer")

        # Display Prediction
        st.success(f"The predicted answer is: **{predicted_answer}**")

# Footer
st.markdown("---")
st.write("Powered by [Streamlit](https://streamlit.io) | Developed with ❤️ by your AI assistant.")
