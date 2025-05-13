import os
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Lambda
from model_classification import MyCNN
from PIL import Image
import matplotlib.pyplot as plt

def resize_pad(image):
    width, height = image.size
    if width > height:
        new_width = 224
        new_height = int(height * (new_width / width))
    else:
        new_height = 224
        new_width = int(width * (new_height / height))
    
    resized_image = image.resize((new_width, new_height))
    
    pad_left = (224 - new_width) // 2
    pad_right = 224 - new_width - pad_left
    pad_top = (224 - new_height) // 2
    pad_bottom = 224 - new_height - pad_top

    padded_image = Image.new("RGB", (224, 224), (0, 0, 0))
    padded_image.paste(resized_image, (pad_left, pad_top))

    return padded_image


def inference():
    # --- Configuration ---
    categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Player number categories
    model_path = "trained_models/best.pt"  # Path to the trained model
    image_path = "1.jpg"  # Path to the image you want to test
    # image_path = "dog.jpeg" # Path to the image you want to test
    # ---------------------

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model ---
    model = MyCNN(num_classes=len(categories))  # Initialize the model
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the trained weights
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # --- Image Preprocessing ---
    transform = Compose([
        Lambda(resize_pad),
        ToTensor(),
    ])

    # --- Load and Preprocess Image ---
    try:
        # image = cv2.imread(image_path)  # OpenCV: channel order is BGR
        # if image is None:
        #     raise FileNotFoundError(f"Could not open or find the image at: {image_path}")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        image = Image.open(image_path).convert("RGB") # read image by PIL
        image_copy = image.copy()
        image = transform(image)
        image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading or preprocessing image: {e}")
        return

    # --- Inference ---
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_number = categories[predicted_class]

    # --- Visualization ---
    plt.figure(figsize=(4, 4))
    plt.imshow(image_copy)
    plt.title(f"Predicted Player Number: {predicted_number}")
    plt.axis("off")
    plt.show()

    print(f"Predicted Player Number: {predicted_number}")

if __name__ == "__main__":
    inference()
