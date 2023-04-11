from PIL import ImageGrab
import numpy as np
from keras.models import load_model
from tkinter import *
import tkinter as tk
import matplotlib.pyplot as plt
from tensorflow import keras 
from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K

########## LOAD THE TEST SET

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_test  = keras.utils.to_categorical(y_test, 10)
x_test  = x_test.astype('float32')
x_test /= 255
print (x_test.shape[0], 'test samples')

########## LOAD THE MODEL

# model = load_model('mnist.h5')
model = load_model('myChatGPTModel.h5')
print("The model was successfully loaded. Now testing!")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:',     score[0])
print('Test accuracy:', score[1])

########## Predict Digit

def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.0
    img = 1 - img
    pic = np.squeeze(img)
    # displayImg (pic)
    # predicting
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

########## Show the digits on the screen

def displayImg (img):
    # print (img)
    plt.subplot(2, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

########## App

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=400, height=400, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.btn_classify = tk.Button(self, text = "Recognise", command =self.classify_handwriting) 
        self.clear_button= tk.Button(self, text = "Clear",command = self.clear_all)
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.btn_classify.grid(row=1, column=1, pady=2, padx=2)
        self.clear_button.grid(row=1, column=0, pady=2)
        self.canvas.bind("<Motion>", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def draw_lines(self, event):
        if event.state == 256: # Check if left mouse button is pressed
            self.x = event.x
            self.y = event.y
            r=8
            self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
    def classify_handwriting(self):
        # Get the coordinates of the canvas
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        bbox=(x, y, x1, y1)
        print ("bbox: ", bbox)
        # Take a screenshot of the canvas and convert to grayscale
        im = ImageGrab.grab(bbox = bbox)
        #im = im.convert('L')
        # Resize the image to 28x28 (the input size of the MNIST model)
        #im = im.resize((28, 28))
        digit, acc = predict_digit(im)
        # print (str(digit)+', '+ str(int(acc*100))+'%')
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')

app = App()
mainloop()
