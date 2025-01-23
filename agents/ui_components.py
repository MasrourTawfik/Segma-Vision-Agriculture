# modules/ui_components.py
import streamlit as st
from typing import List, Dict
from .yield_prediction import CropYieldPredictor

class UIManager:
    def render_header(self):
        st.title("Agricultural Analysis System")
        st.write("An AI-powered tool for agricultural analysis and predictions")

    def render_sidebar(self):
        with st.sidebar:
            st.header("About")
            st.write("This application uses AI to assist with various agricultural analyses.")
            st.markdown("---")
            st.subheader("Available Tasks")
            st.write("• Plant Disease Detection")
            st.write("• Weed Detection")
            st.write("• Soil Classification")
            st.write("• Crop Yield Prediction")
            st.write("• Plant Identification")

    def render_tasks(self, tasks: List[Dict], crop_predictor: CropYieldPredictor):
        for task in tasks:
            if not self._validate_task(task):
                continue
                
            st.subheader(f"Task: {task['task']}")
            
            if task['task'] == "Agriculture crop yield prediction":
                self._render_crop_prediction_ui(crop_predictor)
            elif task['image_required']:
                self._render_image_upload_ui(task)

    def _validate_task(self, task: Dict) -> bool:
        required_keys = ['task', 'image_required', 'additional_info_required']
        return all(key in task for key in required_keys)

    def _render_crop_prediction_ui(self, crop_predictor: CropYieldPredictor):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            city = st.text_input("City name:")
        with col2:
            days = st.number_input("Days to analyze:", min_value=1, value=30)
        with col3:
            crop = st.selectbox("Select crop:", crop_predictor.get_available_crops())
            
        if st.button("Predict Yield"):
            if city and crop:
                with st.spinner("Calculating yield prediction..."):
                    result = crop_predictor.predict(city, crop, days)
                    if result is not None:
                        st.success(f"Predicted yield: {result:.2f}")
                    else:
                        st.error("Unable to make prediction. Please check inputs.")

    def _render_image_upload_ui(self, task: Dict):
        uploaded_file = st.file_uploader("Upload image:", type=['jpg', 'png'])
        if uploaded_file:
            st.image(uploaded_file)
            st.info(f"Image processing for {task['task']} to be implemented")