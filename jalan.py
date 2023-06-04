from flask_cors import CORS

from flask import Flask, request, jsonify
from skimage.io import imread
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np
app = Flask(__name__)
CORS(app)
# load the trained KNN model
knn_model = joblib.load('models.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # get the image from the request
    image = imread(request.files['image'], as_gray=True)

    # convert image to unsigned byte (integer) type
    image = np.uint8(255 * image)

    # calculate GLCM matrix and extract features
    glcm = graycomatrix(image, [5], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # predict the label of the image using the trained KNN model
    predicted_label = knn_model.predict([[contrast, dissimilarity, homogeneity, energy, correlation]])

    # return the predicted label as a JSON response
    return jsonify({'predicted_label': predicted_label.tolist()}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
# from flask_cors import CORS
# from flask import Flask, request, jsonify
# from skimage.io import imread
# from skimage.feature import graycomatrix, graycoprops
# from sklearn.neighbors import KNeighborsClassifier
# import joblib
# import numpy as np
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import firestore

# app = Flask(__name__)
# CORS(app)

# # Initialize Firebase credentials
# cred = credentials.Certificate('C:/Users/LENOVO/Desktop/Klasifikasi_Daging/key.json')
# firebase_admin.initialize_app(cred)

# # Load the trained KNN model
# knn_model = joblib.load('models.joblib')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the image from the request
#     image = imread(request.files['image'], as_gray=True)

#     # Convert image to unsigned byte (integer) type
#     image = np.uint8(255 * image)

#     # Calculate GLCM matrix and extract features
#     glcm = graycomatrix(image, [5], [0], 256, symmetric=True, normed=True)
#     contrast = graycoprops(glcm, 'contrast')[0, 0]
#     dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
#     homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
#     energy = graycoprops(glcm, 'energy')[0, 0]
#     correlation = graycoprops(glcm, 'correlation')[0, 0]

#     # Predict the label of the image using the trained KNN model
#     predicted_label = knn_model.predict([[contrast, dissimilarity, homogeneity, energy, correlation]])

#     # Save the predicted label to Firebase Firestore
#     db = firestore.client()
#     doc_ref = db.collection('predictions').document()
#     doc_ref.set({
#         'predicted_label': predicted_label.tolist()
#     })

#     # Return the predicted label as a JSON response
#     return jsonify({'predicted_label': predicted_label.tolist()}), 200

# if __name__ == '__main__':
#     app.run(port=5500)