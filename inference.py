import os
import shutil
from ultralytics import YOLO
import cv2
from torchvision.transforms import Compose, ToTensor, Lambda
import torch
from model_classification import MyCNN
import matplotlib.pyplot as plt  

from PIL import Image

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

    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Input video path for inference
    input_video_path = os.path.join(project_dir, "data", "football_test", "Match_2031_5_0_test", "Match_2031_5_0_test.mp4")

    # Output video path
    output_dir = os.path.join(project_dir, "output_annotated")

    # Detection model path
    detection_model_path = os.path.join(project_dir, "runs", "train", "weights", "best.pt")

    # Classification model path
    classification_model_path = os.path.join(project_dir, "trained_models", "best.pt")

    # Confidence thresholds
    detection_confidence_ts = 0.6
    classification_confidence_ts = 0.6

    # Player number categories
    categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  

    # Classification transform
    classification_transform = Compose([
        Lambda(resize_pad),
        ToTensor(),
    ])

    if not os.path.exists(detection_model_path):
        print("Error: Detection model not found")
        exit()

    if not os.path.exists(classification_model_path):
        print("Error: Classification model not found")
        exit()

    try:
        detection_model = YOLO(detection_model_path)
    except Exception as e:
        print(f"Error loading detection model: {e}")
        exit()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classification_model = MyCNN(num_classes=12)
        classification_model.load_state_dict(torch.load(classification_model_path, map_location="cpu"))
        classification_model.to(device)
        classification_model.eval()
    except Exception as e:
        print(f"Error loading classification model: {e}")
        exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    try:
        cap = cv2.VideoCapture(input_video_path)
    except Exception as e:
        print(f"Error opening video file: {e}")
        exit()

    if not cap.isOpened():
        print("Could not open video at {}".format(input_video_path))
        exit()

    print("Num of frames: ", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Get video properties for the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create dynamic video output file name
    input_video_filename = os.path.basename(input_video_path)
    input_video_name, input_video_ext = os.path.splitext(input_video_filename)
    output_video_filename = f"{input_video_name}_annotated{input_video_ext}"
    output_video_path = os.path.join(output_dir, output_video_filename)
    
    # Create VideoWriter 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_detection = 0.7 
    font_thickness_detection = 2
    font_scale_classification = 1.5
    font_thickness_classification = 3 

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        annotated_frame = frame.copy()

        # --- Show the frame ---
        # plt.figure(figsize=(10, 6))
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.title(f"Original Frame {frame_count}")
        # plt.axis("off")
        # plt.show()
        # ---------------------------------------------    

        detection_outputs = detection_model.predict(source=frame, conf=detection_confidence_ts, verbose=False)

        player_count = 0  # Counter for player images
        for r in detection_outputs:
            boxes = r.boxes
            for box in boxes:
                bb = box.xyxy[0].int().tolist()
                conf = round(box.conf[0].item(), 2)
                class_id = int(box.cls[0].item())
                class_name = detection_model.names[class_id]

                if class_name == "player":
                    player_image = frame[bb[1]:bb[3], bb[0]:bb[2]]

                    if player_image.size > 0: 
                        player_image_copy = player_image.copy() # Create a copy for visualization

                        player_image = Image.fromarray(cv2.cvtColor(player_image, cv2.COLOR_BGR2RGB)) # Convert to PIL Image
                        player_image = classification_transform(player_image)
                        player_image = player_image.unsqueeze(0).to(device)

                        with torch.no_grad():
                            classification_output = classification_model(player_image)
                            probabilities = torch.softmax(classification_output, dim=1)
                            # print("classification_output: ", classification_output)
                            # print("probabilities: ", probabilities)
                            
                            predicted_class = torch.argmax(classification_output, dim=1).item()
                            predicted_number = categories[predicted_class]

                            confidence = probabilities[0][predicted_class].item()
                            # print("predicted_class: ", predicted_class)
                            # print("confidence: ", confidence)

                            if confidence >= classification_confidence_ts:                            
                                predicted_number_description = "{}".format(predicted_number)
                            else:
                                predicted_number_description = "?"

                            # --- Show the player  ---
                            # player_count += 1
                            # plt.figure(figsize=(4, 4))
                            # plt.imshow(cv2.cvtColor(player_image_copy, cv2.COLOR_BGR2RGB))
                            # plt.title(f"Player {player_count} - Predicted Number: {predicted_number_description}")
                            # plt.axis("off")
                            # plt.show()
                            # -------------------------------------------------------------------                        

                        # Drawing bounding box for player
                        cv2.rectangle(annotated_frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)

                        # annotate detection 
                        label_text_detection = "{}: {:.2f}".format(class_name, conf)
                        cv2.putText(annotated_frame, label_text_detection, (bb[0], bb[1] - 30), font, font_scale_detection, (0, 255, 0), font_thickness_detection)

                        # annotate classification
                        label_text_classification = "{}".format(predicted_number_description)
                        text_size = cv2.getTextSize(label_text_classification, font, font_scale_classification, font_thickness_classification)[0]
                        text_x = bb[0]
                        text_y = bb[3] + text_size[1] + 10
                        cv2.putText(annotated_frame, label_text_classification, (text_x, text_y), font, font_scale_classification, (0, 255, 0), font_thickness_classification)

        out.write(annotated_frame)
        print("Annotated frame {} is written to video".format(frame_count))

        if frame_count == 1000:
            break

    cap.release()
    out.release()
    print("Annotated video is saved to ".format(output_video_path))

if __name__ == "__main__":
    inference()
