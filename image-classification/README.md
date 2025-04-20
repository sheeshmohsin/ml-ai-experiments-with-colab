# 🐱🐶 Cats vs Dogs Image Classifier (TensorFlow + Google Colab)

This project demonstrates how to train a simple Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow and Keras in a Google Colab notebook.

---

## 📂 Project Structure

```
my_dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
```

> Each folder name (`cats/`, `dogs/`) is treated as a class label automatically by `ImageDataGenerator.flow_from_directory()`.

---

## 🚀 Features

- Loads and preprocesses custom image dataset
- Uses `ImageDataGenerator` for normalization and dataset splitting
- Builds a CNN with `Conv2D`, `MaxPooling2D`, `Dropout`, and `Dense` layers
- Trains and evaluates the model on Google Colab with GPU acceleration
- Predicts new unseen images using the trained model

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Google Colab
- Matplotlib (for visualizations)

---

## 🧪 How to Run

1. Upload your image dataset to Google Colab (ZIP with class-wise folders)
2. Unzip and load the data using `ImageDataGenerator`
3. Train the model with `.fit()`
4. Evaluate and visualize performance

---

## 📈 Sample Model Summary

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 128, 128, 32)      896
max_pooling2d (MaxPooling2D) (None, 64, 64, 32)        0
flatten (Flatten)            (None, 131072)            0
dense (Dense)                (None, 128)               16737344
dropout (Dropout)            (None, 128)               0
dense_1 (Dense)              (None, 2)                 258
=================================================================
Total params: 16,738,498
Trainable params: 16,738,498
```

---

## 📸 Prediction Example

```python
from tensorflow.keras.preprocessing import image

img = image.load_img('my_dog.jpg', target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print(f"Predicted class: {predicted_class}")
```

---

## 📌 Notes

- Make sure images are clear, centered, and labeled properly in folders.
- With just a few images, the model may **overfit** — use more data for better generalization.
- You can easily extend this to classify more than two classes.

---

## 📚 License

This project is open source and free to use for learning and experimentation.
