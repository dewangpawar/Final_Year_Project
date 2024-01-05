import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from io import BytesIO
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Define the image dimensions and number of classes
img_width, img_height = 224, 224
num_classes = 2

# Load the saved model
model = tf.keras.models.load_model('Dataset/mri_model_1.h5')

def list_generate(sym):
    list1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    sym = list(filter(lambda x: x != 'default', sym))
    sym = [eval(i) for i in sym]
    for i in sym:
        list1[i] = 1
    return list1

def multidisease_detection(sym_list):
    multidisease_dataset = pd.read_csv('Dataset/Training.csv')
    multidisease_dataset['prognosis'].value_counts()
    multidisease_dataset.groupby('prognosis').mean()
    X = multidisease_dataset.drop(columns = 'prognosis', axis=1)
    Y = multidisease_dataset['prognosis']

    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)

    X = standardized_data
    Y = multidisease_dataset['prognosis']

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=24)

    classifier = svm.SVC(kernel='linear')

    #training the support vector Machine Classifier
    classifier.fit(X_train, Y_train)

    # # accuracy score on the training data
    # X_train_prediction = classifier.predict(X_train)
    # training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    # # print('Accuracy score of the training data : ', training_data_accuracy)

    # # accuracy score on the test data
    # X_test_prediction = classifier.predict(X_test)
    # test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    # # print('Accuracy score of the test data : ', test_data_accuracy)
    # list1 = [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    input_data = (sym_list)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)
    # print(std_data)

    prediction = classifier.predict(std_data)
    print(prediction)

    return prediction

def mri_disease_detection(img_path):
    # Load and preprocess your uploaded image
    img_content = img_path.read()
    img = image.load_img(BytesIO(img_content), target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    # Use your model to predict the class of the uploaded image
    pred = model.predict(img)
    class_idx = np.argmax(pred[0])
    classes = {'Adenocarcinoma Disease': 0, 'COVID-19 Disease': 1, 'Large Cell Carcinoma Disease': 2, 'Normal CT Scan': 3, 'Normal X-ray': 4, 'Pneumonia Disease': 5, 'Squamous Cell Carcinoma Disease': 6}
    class_label = list(classes.keys())[list(classes.values()).index(class_idx)]

    return class_label