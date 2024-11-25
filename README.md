# Dermatological Disease Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-92%25-green)

This repository contains a deep learning-based solution for predicting dermatological diseases using *Convolutional Neural Networks (CNNs)* and *Transfer Learning. The model achieves an impressive **92% accuracy* on a well-structured dataset, showcasing its reliability for disease diagnosis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview

Early detection of dermatological diseases is crucial for effective treatment and management. This project leverages deep learning techniques, particularly CNNs and Transfer Learning, to classify dermatological conditions based on image data. The model has been trained and fine-tuned to deliver high accuracy, making it a potential tool for assisting healthcare professionals.

## Features

- Uses *Convolutional Neural Networks (CNN)* for image classification.
- Incorporates *Transfer Learning* for improved performance.
- Achieves *92% accuracy* on the test dataset.
- User-friendly code structure for easy experimentation and modification.

## Dataset

The dataset used in this project consists of labeled images of various dermatological conditions. 
- *Classes*: Multiple dermatological disease categories ![image](https://github.com/user-attachments/assets/dbc1e245-8080-4e9b-9a40-8c406b6c50ba)

- *Size*: Approximately 3000 images.
- *Preprocessing*: Images were resized, normalized, and augmented to enhance model performance.

> *Note*: The dataset is not included in this repository due to size constraints. You can download the dataset from Internet.

## Technologies Used

- *Python 3.x*
- *TensorFlow/Keras*
- *NumPy*
- *Matplotlib*
- *Pandas*
- *Streamlit*

## Model Architecture

1. *Convolutional Neural Networks (CNN):*
   - Designed a custom CNN for feature extraction and classification.
2. *Transfer Learning*

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/prathamesh177/Derm-AI/tree/main.git

## Model Architecture

![image](https://github.com/user-attachments/assets/20cc2fa0-5ac8-41b5-9192-fa20a79a2775)
![image](https://github.com/user-attachments/assets/f8c04779-b01c-4b68-9d58-a8b330f0174d)


## Results 

![WhatsApp Image 2024-10-27 at 21 41 13_360d1dc3](https://github.com/user-attachments/assets/5a9603cf-bda0-44f9-9c4f-5083f3190621)

## Future Scope
Integration of Multimodal Data:

Incorporating additional data types, such as patient history, genetic factors, and environmental
conditions, could enhance the model’s predictive capabilities, providing a more holistic
diagnostic solution.

Future iterations of the model can integrate explainable AI techniques to make the predictions
interpretable for clinicians. This will build trust in AI systems by providing insights into the
decision-making process of the algorithm.

Continuously updating the model with new data to enhance its predictive capabilities and adapt
to emerging skin disease trends.

Ensuring patient privacy and data security by implementing robust encryption and compliance
with healthcare regulations.

Integrating patient medical history, genetic factors, and environmental exposures to improve
prediction accuracy.
