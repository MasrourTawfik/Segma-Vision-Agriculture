import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

class DiseaseDetector:
    def __init__(self):
        self.model = None
        self.class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Corn___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn___Common_rust',
            'Corn___Northern_Leaf_Blight',
            'Corn___healthy',
            # Add more classes as needed
        ]
        self.img_size = (224, 224)
        self._load_model()

    def _load_model(self):
        """Load the disease detection model"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'plant_disease_model.h5')
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image_data):
        """Preprocess the image for model input"""
        try:
            # Convert bytes to image if needed
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str):
                image = Image.open(image_data)
            else:
                image = Image.open(io.BytesIO(image_data.read()))

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize
            image = image.resize(self.img_size)

            # Convert to array and preprocess
            img_array = tf.keras.preprocessing.image.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            return img_array

        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise

    def detect_disease(self, image_data):
        """Detect disease in the plant image"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_data)

            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])

            # Get the predicted class name
            predicted_class = self.class_names[predicted_class_index]

            # Prepare the result
            result = {
                "disease": predicted_class.replace('___', ' - ').replace('_', ' '),
                "confidence": f"{confidence * 100:.2f}%",
                "is_healthy": "healthy" in predicted_class.lower(),
                "recommendations": self._get_recommendations(predicted_class)
            }

            return result

        except Exception as e:
            print(f"Error in disease detection: {str(e)}")
            raise

    def _get_recommendations(self, disease_class):
        """Get treatment recommendations based on the detected disease"""
        recommendations = {
            'Apple___Apple_scab': [
                "Apply fungicide early in the growing season",
                "Remove infected leaves and fruit",
                "Improve air circulation by proper pruning",
                "Clean up fallen leaves in autumn"
            ],
            'Apple___Black_rot': [
                "Remove infected fruit and cankers",
                "Prune during dry weather",
                "Apply fungicides during the growing season",
                "Maintain good sanitation practices"
            ],
            # Add more recommendations for other diseases
            'healthy': [
                "Continue regular monitoring",
                "Maintain current care practices",
                "Follow preventive care guidelines"
            ]
        }

        return recommendations.get(disease_class, ["Consult a local agricultural expert for specific treatment recommendations"])

def detect_disease(image_file):
    """
    Main function to detect plant diseases
    
    Args:
        image_file: Uploaded file object containing the plant image
        
    Returns:
        dict: Detection results including disease, confidence, and recommendations
    """
    try:
        detector = DiseaseDetector()
        result = detector.detect_disease(image_file)
        return result
        
    except Exception as e:
        raise Exception(f"Error in disease detection: {str(e)}")

if __name__ == "__main__":
    # Example usage
   with open("test_plant.jpg", "rb") as image_file:
            result = detect_disease(image_file)
            print("Detection Result:", result)
