import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
#from geopy.geocoders import Nominatim
#import folium
#from folium import Marker, Popup
#from geopy.geocoders import Nominatim
import requests

# Load the trained model
model = load_model('https://github.com/Sonali-2147/DERM-AI/blob/main/trained_model.h5')

# Define class names
class_names = ["Acne_Rosacea","benign","Eczema","maligant","Melanoma Skin Cancer Nevi and Moles"]
# Define disease names corresponding to class labels
disease_names = ['A preliminary diagnosis for acne and rosacea, two common dermatological conditions, would typically involve assessing the presence of comedones, papules, and pustules for acne, and examining for persistent facial redness, flushing, and visible blood vessels for rosacea.   𝗣𝗥𝗘𝗖𝗔𝗨𝗧𝗜𝗢𝗡𝗔𝗥𝗬 𝗠𝗘𝗔𝗦𝗨𝗥𝗘𝗦:-To manage acne and rosacea, opt for gentle skincare products and avoid harsh cleansers or abrasive scrubs. Always use sunscreen with at least SPF 30 to protect your skin from sun damage, which can exacerbate rosacea symptoms and lead to post-inflammatory hyperpigmentation in acne-prone skin. Additionally, identify and avoid triggers such as spicy foods, alcohol, and stress to minimize flare-ups of both conditions.', 'A preliminary diagnosis for a benign dermatological disease often involves evaluating the appearance, texture, and location of skin lesions, along with considering patient history and risk factors, to differentiate it from potentially more serious conditions.', 'A preliminary diagnosis for eczema involves assessing the presence of characteristic symptoms such as redness, itching, and dry, scaly patches of skin, often in conjunction with a patients medical history and possible triggers.','A preliminary diagnosis for a malignant dermatological disease typically involves identifying concerning features in skin lesions, such as irregular borders, asymmetry, changes in color or size, and history of rapid growth or ulceration, necessitating further evaluation through biopsy or imaging.','A preliminary diagnosis for melanoma skin cancer, nevi, and moles entails examining for suspicious features including asymmetry, irregular borders, uneven color distribution, diameter larger than 6mm, and evolving characteristics, often requiring dermoscopic evaluation and biopsy for confirmation.']  # Update with actual disease names

# Preprocess the input images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match the input size of the model
    img = img / 255.0  # Normalize pixel values between 0 and 1
    return img


# Perform inference
def classify_image(image_path, model):
    # Preprocess the input image
    img = preprocess_image(image_path)
    # Expand dimensions to match the input shape of the model
    img = np.expand_dims(img, axis=0)
    # Perform inference
    predictions = model.predict(img)
    # Get the predicted class label
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_names[predicted_class_index]
    # Map predicted class label to disease name
    disease_name = disease_names[predicted_class_index]
    return predicted_class_label, disease_name

# Display image with predicted class label and diagnosis
def display_predicted_image(image_path, predicted_class, disease_name):
    # Load the image
    img = cv2.imread(image_path)
    # Resize image for display
    img = cv2.resize(img, (700, 700))
    # Overlay predicted class label and diagnosis on the image
    text = f'Predicted class: {predicted_class}'
    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    # Display the image
    st.image(img, caption='Uploaded Image with Prediction', use_column_width=True)

# Streamlit app
def main():

    st.title('DERM-AI')
    # st.title("Nearby Hospitals Map")
    # st.write("Below is a map showing nearby hospitals.")

    # location = get_user_location()
    # if location:
    #     st.write(f"Your current location: {location}")
    #     get_nearby_hospitals(location)
    # else:
    #     st.error("Error fetching location. Please try again later.")


    st.markdown(
        """
        <style>
            .sidebar .sidebar-content {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
            }
            .sidebar .sidebar-content .sidebar-close-button {
                color: #000;
            }
            .reportview-container .main .block-container {
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .reportview-container .main .block-container img {
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title('Menu')
    page = st.sidebar.selectbox('Select a page', ['Home', 'Prediction'])

    if page == 'Home':
        st.write('Welcome to the Derm-AI !!')
        st.write('')
        st.write('How It Works:')
        st.write('1. *Upload Image:* Go to the *Disease Detection* page and upload an image of a skin disease.')
        st.write('2. *Analysis:* Our system will process the image using advanced algorithms to identify potential diseases.')
        st.write('3. *Results:* View the results and recommendations for further action.')

    elif page == 'Prediction':
        st.write('Upload an image to predict the disease.')

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            # Save the uploaded image temporarily
            image_path = 'temp_uploaded_image.jpg'
            with open(image_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            # Perform prediction
            predicted_class_label, disease_name = classify_image(image_path, model)
            # Display predicted image with diagnosis
            display_predicted_image(image_path, predicted_class_label, disease_name)
            st.write('*𝗣𝗿𝗲𝗱𝗶𝗰𝘁𝗲𝗱 𝗰𝗹𝗮𝘀𝘀*', predicted_class_label)
            st.write('*Preliminary Diagnosis:*', disease_name)
            html_code = """
    <div id="map" style="width: 100%; height: 500px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.7.1/dist/leaflet.js"></script>
    <script>
        var map = L.map("map").setView([16.6956, 74.2317], 15);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            maxZoom: 19,
            attribution: "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
        }).addTo(map);
    </script>
    """
            st.components.v1.html(html_code)




if __name__ == '__main__':
    main()
