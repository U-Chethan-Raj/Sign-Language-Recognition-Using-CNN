# Sign-Language-Recognition-Using-Deep-Learning


## Overview
This project aims to classify American Sign Language (ASL) letters using deep learning techniques. The model is trained on the **Sign Language MNIST Dataset**, which consists of labeled images of hand gestures representing different letters. The project leverages **Convolutional Neural Networks (CNNs)** for image classification and evaluates the performance using various machine learning models, including:

- **K-Nearest Neighbors (KNN)**
- **Artificial Neural Networks (ANN)**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Deep Learning Model (CNN)**

## Dataset
The dataset used for training and testing is available at:
- [Sign Language MNIST Train Dataset](https://drive.google.com/file/d/1BMdIGWPZZQCHmpOW9Tm6TMSomqpyTuJF/view?usp=drive_link)
- [Sign Language MNIST Test Dataset](https://drive.google.com/file/d/1Xw2CmB-Ais7RlzhyEtNT8Mu46gy8Ztlz/view?usp=drive_link)

## Dependencies
Ensure you have the following libraries installed:
```bash
pip install numpy pandas tensorflow matplotlib seaborn scikit-learn opencv-python
```

## Project Structure
- **Data Preprocessing**:
  - Load the dataset using Pandas.
  - Reshape and normalize the images.
  - Visualize the first 10 images.
- **Data Augmentation**:
  - Use `ImageDataGenerator` to apply transformations like rotation, shifting, zooming, and flipping.
- **Model Training**:
  - Define a **CNN architecture** with convolutional and pooling layers.
  - Compile and train the model using `Adam` optimizer and `SparseCategoricalCrossentropy` loss.
- **Evaluation & Visualization**:
  - Plot accuracy and loss curves.
  - Compute precision, recall, and F1-score.
  - Display the confusion matrix.
- **Sign Language Prediction**:
  - Load and preprocess an image.
  - Predict the corresponding letter.

## Model Architecture (CNN)
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])
```

## Training Process
- The model is compiled with:
  ```python
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
  )
  ```
- Trained for **10 epochs** using `fit_generator()`.

## Performance Metrics
- The trained model achieves high accuracy on validation data.
- Performance is evaluated using a confusion matrix and classification report.

## Prediction on New Images
You can use the trained model to predict ASL letters from new images:
```python
def predict_sign_language(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.reshape(-1, 28, 28, 1)
    image = image.astype('float32') / 255.0
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    sign_label = chr(predicted_class + 65)
    print("Predicted sign label:", sign_label)
```

## Conclusion
This project successfully implements deep learning and machine learning models to classify American Sign Language letters. The CNN model provides the best performance compared to traditional ML algorithms. Future work can involve increasing the dataset size and improving model generalization.

## References
- TensorFlow documentation: https://www.tensorflow.org/
- Scikit-learn: https://scikit-learn.org/stable/
- OpenCV: https://opencv.org/

