from ultralytics import YOLO
import torch
import os
from ultralytics import settings

def train():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    project_name=os.path.join(project_dir, "runs")
    last_model_path = os.path.join(project_name, "train8", "weights", "last.pt")
    data_yaml_path = os.path.join(project_dir, "data", "football_dataset", "football.yaml")
    
    num_epochs = 10
    batch_size = 16
    imgsz = 1024
    workers = 4

    if resume == False and os.path.exists(last_model_path):
        print(f"Resuming training from: {last_model_path}")
        try:
            model = YOLO(last_model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return 
    elif resume == False:
        print(f"Staring training from scratch")
        try:
            model = YOLO("yolo11n.pt")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    elif resume == True and os.path.exists(last_model_path):
        print(f"Resuming training from: {last_model_path}")
        try:
            model = YOLO(last_model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return 
    elif resume == True: 
        print(f"Error: Could not find last checkpoint at: {last_model_path}")

    if torch.cuda.is_available():
        device = 0  # Use GPU 0
        print("CUDA is available! Training on GPU.")
    else:
        device = "cpu"  # Use CPU
        print("CUDA is NOT available. Training on CPU.")

    try:
        model.train(
            data=data_yaml_path,
            epochs=num_epochs,
            batch=batch_size,  
            imgsz=imgsz,
            device=device,
            workers=workers,
            project = project_name,
            resume = resume
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        exit()

if __name__ == "__main__":
    resume = False
    train()





