from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import keras
from keras import layers
from keras.models import Sequential

import openai
openai.api_key = "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"
engine="text-curie-001" 

import pathlib
batch_size = 32
img_height = 180
img_width = 180
class_names = ['beef tartare', 'beet salad', 'beignets', 'caesar salad', 'cheesecake', 'chicken quesadilla', 'chicken wings', 'chocolate cake', 'club sandwich', 'cup cakes', 'donuts', 'dumplings', 'fish and chips', 'french fries', 'french toast', 'fried calamari', 'fried rice', 'greek salad', 'grilled cheese sandwich', 'hamburger', 'hot dog', 'ice cream', 'omelette', 'pancakes', 'pizza', 'pork chop', 'spaghetti carbonara', 'steak', 'sushi', 'tacos', 'tiramisu', 'waffles']

num_classes = len(class_names)

model = keras.models.load_model(r'net')

def ask(prompt):
    completion = openai.Completion.create(engine="text-davinci-003", 
                                          prompt=prompt, 
                                          temperature=0.5, 
                                          max_tokens=1000)
    return(completion.choices[0]['text'])

class App:
    def __init__(self, master):
        self.master = master
        master.title("My App")

        self.image_label = Label(master)
        self.image_label.pack(side=LEFT)

        self.text_scrollbar = Scrollbar(master)
        self.text_scrollbar.pack(side=RIGHT, fill=Y)

        self.text = Text(master, yscrollcommand=self.text_scrollbar.set)
        self.text.pack(side=RIGHT, fill=BOTH)
        self.text_scrollbar.config(command=self.text.yview)

        self.button_frame = Frame(master)
        self.button_frame.pack(side=BOTTOM)

        self.choose_image_button = Button(self.button_frame, text="Choose Image", command=self.choose_image)
        self.choose_image_button.pack(side=LEFT)

        self.clear_button = Button(self.button_frame, text="Clear", command=self.clear)
        self.clear_button.pack(side=RIGHT)
    
    def analise(self, img):
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        str = class_names[np.argmax(score)]
        return str

    def choose_image(self):
        # Запрашиваем у пользователя путь к файлу картинки
        file_path = filedialog.askopenfilename()

        # Если пользователь выбрал файл, загружаем изображение и выводим его
        if file_path:
            image = Image.open(file_path)
            image = image.resize((400, 400)) # уменьшаем размер изображения
            img = image.resize((180, 180))
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)
            self.image_label.pack()

            strok = self.analise(img) + ' ' + ask('Назови рецепт ' + self.analise(img))

            # Выводим текст в текстовом поле
            
            self.text.delete(1.0, tk.END)
            self.text.insert(tk.END, strok)

    def clear(self):
        self.image_label.config(image=None)
        self.photo = None
        self.text.delete(1.0, END)

root = Tk()
app = App(root)
root.mainloop()