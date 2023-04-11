import numpy as np
from keras.models import load_model
from keras.datasets import mnist

# Load the trained model
model = load_model('myChatGPTModel.h5')

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (test_data, test_label) = mnist.load_data()

# Normalize the test data
test_data = test_data.astype('float32') / 255

test_data   = test_data.reshape(test_data.shape[0], 28, 28, 1)
predictions = model.predict(test_data)
for i in range (10):
    print("Label: " + str(test_label[i]) + " Rec: " + str(np.argmax(predictions[i])))
