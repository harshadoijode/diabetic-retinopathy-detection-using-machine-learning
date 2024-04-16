import os
import tkinter as tk

from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image

import cv2

from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
 # TensorFlow is required for Keras to work
# initialise GUI
np.set_printoptions(suppress=True)
top = tk.Tk()
top.geometry('800x600')
top.title('Diabetic retinopati detection')
img= PhotoImage(file='1.png', master=top)
img_label= Label(top,image=img)

#define the position of the image
img_label.place(x=0, y=0)

def classify(file_path):
    #imageCurrency.runImageClassification(file_path)
    # Load the model
    model = tf.keras.models.load_model('keras_Model.h5')

    # Load the label names
    with open('labels.txt', 'r') as f:
        label_names = f.read().splitlines()

    # Load and preprocess the image
    img_path = file_path
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Predict the class of the image
    predictions = model.predict(x)
    class_index = np.argmax(predictions[0])
    class_name_predicted = label_names[class_index]

    # Print the predicted class
    print('Predicted class:', class_name_predicted)
    label.configure(foreground='#011638', text=class_name_predicted)
    label.place(relx=0.70, rely=0.20)
    elabel = class_name_predicted

    cv2.putText(top, elabel,  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25),(top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

label = Label(top)
label = Label(top, background='white', font=('arial', 15, 'bold'))
sign_image = Label(top)

upload = Button(top, text="Upload an image",  padx=10, pady=5)
#webcam = Button(top, text="Live Mushroom Detection", command=liveEmotion, padx=21, pady=5)
upload.configure(background='#364156', foreground='white', command=upload_image, font=('arial', 10, 'bold'))
#webcam.pack(side=BOTTOM,pady=70)
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Diabetic retinopati detection", pady=20, font=('arial', 20, 'bold'))

heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()