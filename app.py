import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

class NeuralNetwork:
    def __init__(self, path):
        self.model = models.load_model(path)
        self.model.summary()
    
    def query(self, inputs_list):
        inputs = inputs_list.reshape(1, 784, 1)
        output = self.model.predict(inputs)
        return output

class Tree:
    def __init__(self):
        cities = {  "32": {"Name": "Львів"},
                    "44": {"Name": "Київ"},
                    "48": {"Name": "Одеса"}}

        self.root = {
            "380": {
                "Name": "Україна",
                "Values": {
                "68": {
                    "Name": "Київстар",
                    "Values": cities
                },
                "67": {
                    "Name": "Київстар",
                    "Values": cities
                },
                "96": {
                    "Name": "Київстар",
                    "Values": cities
                },
                "97": {
                    "Name": "Київстар",
                    "Values": cities
                },
                "98": {
                    "Name": "Київстар",
                    "Values": cities
                },
                "63": {
                    "Name": "Lifecell",
                    "Values": cities
                },
                "73": {
                    "Name": "Lifecell",
                    "Values": cities
                },
                "93": {
                    "Name": "Lifecell",
                    "Values": cities
                },
                "50": {
                    "Name": "Vodafone",
                    "Values": cities
                },
                "66": {
                    "Name": "Vodafone",
                    "Values": cities
                },
                "95": {
                    "Name": "Vodafone",
                    "Values": cities
                },
                "99": {
                    "Name": "Vodafone",
                    "Values": cities
                }}
            },
            "1": {"Name": "Канада"},
            "420": {"Name": "Чехія"}
        }

    def predict(self, phone_number, root=None):
        if root is None:
            root = self.root
        result = ""
        for key, node in root.items():
            if key == "Name":
                continue

            if phone_number.startswith(key):
                phone_number = phone_number[len(key):]
                result += " " + node["Name"] + " "
                if len(node) > 1:
                    result += self.predict(phone_number, node.get("Values"))

        return result

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phone number")
        
        self.canvas = tk.Canvas(self.root, width=800, height=400)
        self.canvas.pack()
        
        self.rect = None
        self.start_x = None
        self.start_y = None
        
        self.load_button = tk.Button(self.root, text="Upload Image", command=self.load_image)
        self.load_button.pack()
        
        self.rect_button = tk.Button(self.root, text="Draw Rectangle", command=self.draw_rectangle)
        self.rect_button.pack()
        
        self.clear_button = tk.Button(self.root, text="Clear Rectangle", command=self.clear_rectangle)
        self.clear_button.pack()
        
        self.scan_button = tk.Button(self.root, text="Scan", command=self.scan_image)
        self.scan_button.pack()
        
        self.image = None
        self.image_path = None

        self.result_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.pack()

        self.input_label = tk.Label(self.root, text="Enter Number:")
        self.input_label.pack()
        
        self.input_entry = tk.Entry(self.root)
        self.input_entry.pack()
        
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_number)
        self.predict_button.pack()

    def predict_number(self):
        number_input = self.input_entry.get()
        if number_input.isdigit() and len(number_input) == 12:
            result = tree.predict(number_input)
            self.result_label.config(text=result)
        else: 
            self.result_label.config(text="Please enter a valid number.")

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.image = Image.open(self.image_path)
            self.image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def draw_rectangle(self):
        if self.image:
            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_move_press)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def clear_rectangle(self):
        self.canvas.delete(self.rect)
        self.rect = None

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y

        if self.rect:
            self.canvas.delete(self.rect)

        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red', width=2)

        for i in range(1, 12):
            x = self.start_x + (i * (event.x - self.start_x) // 12)
            self.canvas.create_line(x, self.start_y, x, event.y, fill='blue')

    def on_move_press(self, event):
        cur_x, cur_y = event.x, event.y

        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

        self.canvas.delete("lines")
        for i in range(1, 12):
            x = self.start_x + (i * (cur_x - self.start_x) // 12)
            self.canvas.create_line(x, self.start_y, x, cur_y, fill='blue', tags="lines")

    def on_button_release(self, event):
        pass

    def scan_image(self):
        if self.image:
            x1, y1, x2, y2 = self.canvas.coords(self.rect)
            pil_image = Image.open(self.image_path).convert('L')
            cropped_image = pil_image.crop((x1, y1, x2, y2))
            resized_image = cropped_image.resize((28 * 12, 28))
            cut_images = []
            for i in range(12):
                cut_image = resized_image.crop((i*28, 0, (i + 1)*28, 28))
                img_data  = 255 - np.array(cut_image).reshape(784)
                img_data = (img_data / 255.0 * 0.99) + 0.01
                cut_images.append(img_data)
            
            output_string = ""
            for cut_image in cut_images:
                outputs = nn.query(cut_image)
                predicted_digit = np.argmax(outputs, axis=-1)[0]
                output_string += str(predicted_digit)

            print("Output String:", output_string)
            result = tree.predict(output_string)
            self.result_label.config(text=result)
            
if __name__ == "__main__":
    nn = NeuralNetwork('model.keras')

    tree = Tree()

    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
