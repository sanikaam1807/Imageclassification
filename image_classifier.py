"""
Task 3: Image Classification Using a Pretrained Model
-----------------------------------------------------
Uses MobileNetV2 (pretrained on ImageNet, 1000 classes) via TensorFlow/Keras
to classify a few sample images and display the predicted labels.

Setup:
    pip install tensorflow pillow matplotlib numpy

How to run:
    1. Create a folder named "images" next to this file.
    2. Put 3-5 sample images (.jpg / .png) inside it
       (e.g. a dog, a cat, a car, a banana, a laptop).
    3. Run:  python image_classifier.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# ---------------------------------------------------------------
# Step 1: Load the pretrained model
# weights="imagenet" downloads a model already trained on 1.4M images.
# We don't train anything ourselves - we just use it to predict.
# ---------------------------------------------------------------
print("Loading pretrained MobileNetV2 model...")
model = MobileNetV2(weights="imagenet")

IMAGE_FOLDER = "images"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def classify_image(img_path):
    """Load one image, prepare it, and return the top 3 predictions."""
    # MobileNetV2 expects 224 x 224 pixel images
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert image to a numpy array, then add a batch dimension
    # (the model expects shape: (1, 224, 224, 3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Scale pixel values the same way the model was trained
    img_array = preprocess_input(img_array)

    # Predict: gives 1000 probabilities (one per ImageNet class)
    predictions = model.predict(img_array, verbose=0)

    # Convert the probabilities to readable labels (top 3)
    return img, decode_predictions(predictions, top=3)[0]


def main():
    if not os.path.isdir(IMAGE_FOLDER):
        print(f'Please create a folder named "{IMAGE_FOLDER}" and add some images.')
        return

    files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(VALID_EXTENSIONS)]
    if not files:
        print("No images found in the folder.")
        return

    # Set up a row of plots, one per image
    fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 5))
    if len(files) == 1:
        axes = [axes]

    for ax, filename in zip(axes, files):
        path = os.path.join(IMAGE_FOLDER, filename)
        img, top3 = classify_image(path)

        print(f"\nImage: {filename}")
        for _, label, score in top3:
            print(f"   {label}: {score * 100:.2f}%")

        best_label, best_score = top3[0][1], top3[0][2]
        ax.imshow(img)
        ax.set_title(f"{best_label}\n({best_score * 100:.1f}%)")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
