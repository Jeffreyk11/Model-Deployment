import streamlit as st
import joblib
import numpy as np

# Load the trained machine learning model
@st.cache_resource
def load_model():
    try:
        return joblib.load('trained_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def main():
    st.title('🌿 Iris Flower Prediction App')

    st.write("Adjust the sliders to input feature values for prediction.")

    # User inputs for the 4 features
    sepal_length = st.slider('Sepal Length', min_value=0.0, max_value=10.0, value=5.1)
    sepal_width = st.slider('Sepal Width', min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.slider('Petal Length', min_value=0.0, max_value=10.0, value=1.4)
    petal_width = st.slider('Petal Width', min_value=0.0, max_value=10.0, value=0.2)
    
    if st.button('Make Prediction'):
        if model:
            features = [sepal_length, sepal_width, petal_length, petal_width]
            result = make_prediction(features)
            st.success(f'🌼 The predicted flower species is: **{result}**')
        else:
            st.error("Model not loaded. Please check the trained_model.pkl file.")

def make_prediction(features):
    """Make a prediction using the trained model."""
    input_array = np.array(features).reshape(1, -1)
    try:
        prediction = model.predict(input_array)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error"

if __name__ == '__main__':
    main()
