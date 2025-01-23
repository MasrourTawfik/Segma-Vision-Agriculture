import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import joblib

class SoilClassifier:
    def __init__(self):
        self.image_model = None
        self.data_model = None
        self.img_size = (224, 224)
        self.soil_types = [
            'Clay',
            'Sandy',
            'Loamy',
            'Black',
            'Red',
            'Silty'
        ]
        self._load_models()

    def _load_models(self):
        """Load both image-based and data-based classification models"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
            
            # Load image-based model
            image_model_path = os.path.join(model_path, 'soil_image_model.h5')
            self.image_model = tf.keras.models.load_model(image_model_path)
            
            # Load data-based model
            data_model_path = os.path.join(model_path, 'soil_data_model.pkl')
            self.data_model = joblib.load(data_model_path)
            
            # Load scalers
            self.data_scaler = joblib.load(os.path.join(model_path, 'soil_data_scaler.pkl'))
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
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

    def classify_soil_image(self, image_data):
        """Classify soil type based on image"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_data)

            # Make prediction
            predictions = self.image_model.predict(processed_image)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])

            # Get the predicted soil type
            predicted_soil = self.soil_types[predicted_class_index]

            # Prepare result
            result = {
                "soil_type": predicted_soil,
                "confidence": f"{confidence * 100:.2f}%",
                "characteristics": self._get_soil_characteristics(predicted_soil),
                "recommendations": self._get_recommendations(predicted_soil)
            }

            return result

        except Exception as e:
            print(f"Error in soil classification: {str(e)}")
            raise

    def classify_soil_data(self, soil_data):
        """Classify soil type based on provided data"""
        try:
            # Prepare input data
            input_features = ['ph', 'organic_matter', 'nitrogen', 'phosphorus']
            input_data = np.array([[
                soil_data['ph'],
                soil_data['organic_matter'],
                soil_data['nitrogen'],
                soil_data['phosphorus']
            ]])

            # Scale the input data
            scaled_data = self.data_scaler.transform(input_data)

            # Make prediction
            predicted_class_index = self.data_model.predict(scaled_data)[0]
            prediction_proba = self.data_model.predict_proba(scaled_data)[0]
            confidence = float(prediction_proba[predicted_class_index])

            # Get the predicted soil type
            predicted_soil = self.soil_types[predicted_class_index]

            # Prepare result
            result = {
                "soil_type": predicted_soil,
                "confidence": f"{confidence * 100:.2f}%",
                "characteristics": self._get_soil_characteristics(predicted_soil),
                "recommendations": self._get_recommendations(predicted_soil),
                "analysis": self._analyze_soil_parameters(soil_data)
            }

            return result

        except Exception as e:
            print(f"Error in soil data classification: {str(e)}")
            raise

    def _get_soil_characteristics(self, soil_type):
        """Get characteristics for the identified soil type"""
        characteristics = {
            'Clay': [
                "High water holding capacity",
                "Poor drainage",
                "High nutrient content",
                "Dense and heavy texture"
            ],
            'Sandy': [
                "Low water holding capacity",
                "Excellent drainage",
                "Low nutrient content",
                "Light and loose texture"
            ],
            'Loamy': [
                "Balanced water holding capacity",
                "Good drainage",
                "High nutrient content",
                "Ideal for most crops"
            ],
            'Black': [
                "High organic matter content",
                "Good water retention",
                "Rich in nutrients",
                "Excellent for crop growth"
            ],
            'Red': [
                "High iron content",
                "Good drainage",
                "Moderate fertility",
                "Common in tropical regions"
            ],
            'Silty': [
                "High water holding capacity",
                "Moderate drainage",
                "Good nutrient content",
                "Smooth texture"
            ]
        }
        return characteristics.get(soil_type, ["Characteristics not available"])

    def _get_recommendations(self, soil_type):
        """Get recommendations based on soil type"""
        recommendations = {
            'Clay': [
                "Add organic matter to improve drainage",
                "Avoid working soil when wet",
                "Consider raised beds for better drainage",
                "Choose crops that tolerate clay soil"
            ],
            'Sandy': [
                "Add organic matter to improve water retention",
                "Use mulch to reduce water evaporation",
                "Apply fertilizers in small, frequent doses",
                "Choose drought-tolerant crops"
            ],
            'Loamy': [
                "Maintain organic matter content",
                "Practice crop rotation",
                "Use cover crops",
                "Regular soil testing"
            ],
            'Black': [
                "Maintain organic matter levels",
                "Monitor water content",
                "Practice sustainable farming",
                "Regular nutrient testing"
            ],
            'Red': [
                "Add organic matter",
                "Monitor pH levels",
                "Use appropriate fertilizers",
                "Consider iron-tolerant crops"
            ],
            'Silty': [
                "Improve drainage if needed",
                "Add organic matter",
                "Avoid compaction",
                "Use appropriate tillage practices"
            ]
        }
        return recommendations.get(soil_type, ["Consult a local agricultural expert for specific recommendations"])

    def _analyze_soil_parameters(self, soil_data):
        """Analyze soil parameters and provide insights"""
        analysis = []
        
        # pH analysis
        ph = soil_data['ph']
        if ph < 6.0:
            analysis.append("Soil is acidic - Consider liming")
        elif ph > 7.5:
            analysis.append("Soil is alkaline - Consider adding sulfur")
        else:
            analysis.append("pH is in optimal range")

        # Organic matter analysis
        organic_matter = soil_data['organic_matter']
        if organic_matter < 3:
            analysis.append("Low organic matter - Add compost or organic amendments")
        elif organic_matter > 6:
            analysis.append("High organic matter - Good for most crops")

        # Nutrient analysis
        nitrogen = soil_data['nitrogen']
        if nitrogen < 50:
            analysis.append("Low nitrogen - Consider nitrogen fertilization")
        
        phosphorus = soil_data['phosphorus']
        if phosphorus < 30:
            analysis.append("Low phosphorus - Consider phosphate fertilization")

        return analysis

def classify_soil(image_file=None, soil_data=None):
    """
    Main function to classify soil
    
    Args:
        image_file: Optional uploaded file object containing the soil image
        soil_data: Optional dictionary containing soil test data
        
    Returns:
        dict: Classification results including soil type, characteristics, and recommendations
    """
    try:
        classifier = SoilClassifier()
        
        if image_file is not None:
            return classifier.classify_soil_image(image_file)
        elif soil_data is not None:
            return classifier.classify_soil_data(soil_data)
        else:
            raise ValueError("Either image_file or soil_data must be provided")
        
    except Exception as e:
        raise Exception(f"Error in soil classification: {str(e)}")

if __name__ == "__main__":
    # Example usage for image-based classification
    try:
        with open("test_soil.jpg", "rb") as image_file:
            result = classify_soil(image_file=image_file)
            print("Image Classification Result:", result)
    except Exception as e:
        print(f"Error: {str(e)}")

    # Example usage for data-based classification
    try:
        soil_data = {
            'ph': 6.5,
            'organic_matter': 4.0,
            'nitrogen': 60,
            'phosphorus': 40
        }
        result = classify_soil(soil_data=soil_data)
        print("Data Classification Result:", result)
    except Exception as e:
        print(f"Error: {str(e)}")