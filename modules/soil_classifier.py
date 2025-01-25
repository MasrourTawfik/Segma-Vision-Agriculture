# soil_classifier.py (modules/soil_classifier.py)
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import google.generativeai as genai
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key='AIzaSyCIg8ie1z9mhMBVYY7tq1PY8iBl9vgkYfc')
model_gemini = genai.GenerativeModel('gemini-pro')

class Type(nn.Module):
    def __init__(self, input=16*3*3, output=5):
        super(Type,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=2, padding=1, stride=2)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding=1, stride=2)
        self.pooling = nn.MaxPool2d(kernel_size=2, padding=1, stride=2)
        self.flat = lambda x: x.view(-1, 16*3*3)
        self.lay1 = nn.Linear(input, input*2)
        self.lay2 = nn.Linear(input*2, input*2)
        self.out = nn.Linear(input*2, output)
    
    def forward(self, x):
        x = torch.relu(self.pooling(self.cnn1(x)))
        x = torch.relu(self.pooling(self.cnn2(x)))
        x = self.flat(x)
        x = torch.relu(self.lay1(x))
        x = torch.relu(self.lay2(x))
        x = self.out(x)
        return x

@st.cache_resource
def load_model(model_path: str) -> nn.Module:
    try:
        model = Type()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict_soil_type(
    model: nn.Module, 
    image_tensor: torch.Tensor,
    classes: List[str]
) -> str:
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            return classes[predicted.item()]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def get_soil_recommendations(soil_type: str) -> str:
    try:
        prompt = f"""
        For soil type '{soil_type}', provide:
        1. Top 3 suitable crops
        2. Key soil management practices
        3. Ideal pH range and nutrient requirements
        Limit response to these points only.
        """
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return "Unable to fetch recommendations at this time."

def run_classification():
    st.title("Soil Type Classification and Recommendations")
    st.write("Upload a soil image for classification and recommendations.")

    uploaded_file = st.file_uploader(
        "Upload soil image (JPG, JPEG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Analyzing soil type..."):
                model_path = "models/soil_classification_model.pth"
                model = load_model(model_path)
                
                if model is None:
                    st.error("Failed to load model")
                    return

                classes = [
                    "Black Soil", "Cinder Soil", "Laterite Soil",
                    "Peat Soil", "Yellow Soil"
                ]

                image_tensor = preprocess_image(image)
                soil_type = predict_soil_type(model, image_tensor, classes)

                if soil_type:
                    st.success(f"Predicted Soil Type: {soil_type}")

                    with st.spinner("Generating recommendations..."):
                        recommendations = get_soil_recommendations(soil_type)
                        st.write("### Recommendations")
                        st.write(recommendations)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")