from flask_cors import CORS

from flask import Flask, request, jsonify
from skimage.io import imread
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np
app = Flask(__name__)
CORS(app)

knn_model = joblib.load('models.joblib')

@app.route('/predict', methods=['POST'])
def predict():
  
    image = imread(request.files['image'], as_gray=True)

   
    image = np.uint8(255 * image)

 
    glcm = graycomatrix(image, [5], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

   
    predicted_label = knn_model.predict([[contrast, dissimilarity, homogeneity, energy, correlation]])

    return jsonify({'predicted_label': predicted_label.tolist()}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
