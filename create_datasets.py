import os
import cv2
import json
import random
import shutil
from typing import Final
from tqdm import tqdm

MAX_FRAMES: Final = 0

def process_classification_data(input_path, output_path, split_ratio=0.8):
    print("*** Process create classification dataset to: ", output_path)
    football_dir = output_path
    all_dir = os.path.join(football_dir, "all")
    train_dir = os.path.join(football_dir, "train")
    val_dir = os.path.join(football_dir, "val")

    os.makedirs(football_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    class_dirs = [os.path.join(all_dir, str(i)) for i in range(12)]
    for class_dir in class_dirs:
        os.makedirs(class_dir, exist_ok=True)
    
    sub_dirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    for sub_dir in tqdm(sub_dirs, desc="Processing subdirectories"):
        sub_dir_path = os.path.join(input_path, sub_dir)
        print("Sub dir path: ", sub_dir_path)

        if os.path.isdir(sub_dir_path):
            for file in os.listdir(sub_dir_path):
                if file.endswith(".mp4"):
                    video_path = os.path.join(sub_dir_path, file)
                elif file.endswith(".json"):
                    json_path = os.path.join(sub_dir_path, file)

            with open(json_path, "r") as f:
                annotations = json.load(f)

            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                frame_annotations = [ann for ann in annotations["annotations"] if ann["image_id"] == frame_count]
                for annotation in frame_annotations:
                    if annotation["category_id"]  == 4:
                        x_min, y_min, width, height = annotation["bbox"]
                        x_max = x_min + width
                        y_max = y_min + height
                        player_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

                        if player_image.size > 0:
                            player_label = 0
                            if "attributes" in annotation:
                                if "number_visible" in annotation["attributes"]:
                                        number_visible = annotation["attributes"]["number_visible"]
                                        if number_visible in ["invisible"]:
                                            player_label = 0
                                        elif "jersey_number" in annotation["attributes"]:
                                            jersey_number = int(annotation["attributes"]["jersey_number"])
                                            if 1 <= jersey_number <= 10:
                                                player_label = jersey_number
                                            elif jersey_number >= 11:
                                                player_label = 11

                                class_dir = os.path.join(all_dir, str(player_label))
                                image_filename = f"{sub_dir}_frame_{frame_count}_player_{annotation["id"]}.jpg"
                                image_path = os.path.join(class_dir, image_filename)
                                cv2.imwrite(image_path, player_image)
                            
                if MAX_FRAMES  > 0 and frame_count >= MAX_FRAMES: 
                    break

            print("Total frames: ", frame_count)
            cap.release()

        if MAX_FRAMES  > 0 and frame_count >= MAX_FRAMES: 
            break

    # Split data into train and val 
    for class_label in range(12):
        class_dir = os.path.join(all_dir, str(class_label))
        image_files = os.listdir(class_dir)
        
        random.shuffle(image_files)

        train_size = int(len(image_files) * split_ratio)
        train_files = image_files[:train_size]
        val_files = image_files[train_size:]

        destination_dir = os.path.join(train_dir, str(class_label))
        os.makedirs(destination_dir, exist_ok=True)
        for file in train_files:
            source_path = os.path.join(class_dir, file)
            destination_path = os.path.join(destination_dir, file)
            shutil.move(source_path, destination_path)

        destination_dir = os.path.join(val_dir, str(class_label))
        os.makedirs(destination_dir, exist_ok=True)
        for file in val_files:
            source_path = os.path.join(class_dir, file)
            destination_path = os.path.join(destination_dir, file)
            shutil.move(source_path, destination_path)

    # Delete the all dir 
    for class_label in range(12):
        class_dir = os.path.join(all_dir, str(class_label))
        shutil.rmtree(class_dir)
    shutil.rmtree(all_dir)

def process_detection_data(input_path, output_path, split_ratio=0.8):
    print("*** Process create detection dataset to: ", output_path)

    sub_dirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]

    all_images_dir = os.path.join(output_path, "all_images")
    all_labels_dir = os.path.join(output_path, "all_labels")
    os.makedirs(all_images_dir, exist_ok=True)
    os.makedirs(all_labels_dir, exist_ok=True)

    image_files = []
    for sub_dir in tqdm(sub_dirs, desc="Processing subdirectories"):
        subdir_path = os.path.join(input_path, sub_dir)
        print("Sub dir path: ", subdir_path)

        video_file_path = None
        json_file_path = None

        for file in os.listdir(subdir_path):
            if file.endswith(".mp4"):
                video_file_path = os.path.join(subdir_path, file)
            elif file.endswith(".json"):
                json_file_path = os.path.join(subdir_path, file)

        data = None
        if json_file_path:
            try:
                with open(json_file_path, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print("Json file not found at {}".format(json_file_path))
                continue
            except json.JSONDecodeError:
                print("Invalid Json file at {}".format(json_file_path))
                continue
            
        if video_file_path:
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                print("Could not open video at {}".format(video_file_path))
            else:
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: 
                        break

                    frame_count += 1
                    frame_filename = f"{sub_dir}_frame_{frame_count}.jpg"
                    frame_filepath = os.path.join(all_images_dir, frame_filename)
                    label_filename = frame_filename.replace(".jpg", ".txt")
                    label_filepath = os.path.join(all_labels_dir, label_filename)

                    if os.path.exists(label_filepath):
                        os.remove(label_filepath)

                    has_ball_or_player = False
                    if data:
                        image_id = None
                        for image in data["images"]:
                            if frame_count == image["id"]:
                                image_id = image["id"]
                                image_width = image["width"]
                                image_height = image["height"]
                                break
                        
                        if image_id is not None:
                            for annotation in data["annotations"]:
                                if image_id == annotation["image_id"]:
                                    category_id = annotation["category_id"]
                                    if category_id in [3, 4]:
                                        has_ball_or_player = True
                                        bbox = annotation["bbox"]
                                        
                                        x_min, y_min, width, height = bbox
                                        x_center = x_min + width / 2
                                        y_center = y_min + height / 2

                                        x_center /= image_width
                                        y_center /= image_height
                                        width /= image_width
                                        height /= image_height
                                    
                                        with open(label_filepath, "a") as f:
                                            f.write("{:d} {:.6f} {:.6f} {:.6f} {:.6f}".format(category_id - 3, x_center, y_center, width, height))
                                            f.write("\n")
                    
                            if has_ball_or_player:
                                cv2.imwrite(frame_filepath, frame)
                                image_files.append(frame_filename)

                                if MAX_FRAMES  > 0 and frame_count >= MAX_FRAMES: 
                                    break

            print("Total frames: ", frame_count)
            cap.release()

        if MAX_FRAMES  > 0 and frame_count >= MAX_FRAMES: 
            break

    train_val_ratio = split_ratio
    image_train_dir = os.path.join(output_path, "images", "train")
    image_val_dir = os.path.join(output_path, "images", "val")
    label_train_dir = os.path.join(output_path, "labels", "train")
    label_val_dir = os.path.join(output_path, "labels", "val")

    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    random.shuffle(image_files)

    train_size = int(len(image_files) * train_val_ratio)
    train_image_files = image_files[:train_size]
    val_image_files = image_files[train_size:]

    for file in train_image_files:
        shutil.move(os.path.join(all_images_dir, file), image_train_dir)
        label_file_name = file.replace(".jpg", ".txt")
        shutil.move(os.path.join(all_labels_dir, label_file_name), label_train_dir)

    for file in val_image_files:
        shutil.move(os.path.join(all_images_dir, file), image_val_dir)
        label_file_name = file.replace(".jpg", ".txt")
        shutil.move(os.path.join(all_labels_dir, label_file_name), label_val_dir)

    # Delete all_images_dir and all_labels_dir
    shutil.rmtree(all_images_dir)
    shutil.rmtree(all_labels_dir)

    # Create yaml file
    yaml_file_name = "football.yaml"
    yaml_file_path = os.path.join(output_path, yaml_file_name)
    num_classes = 2
    name_classes = ["ball", "player"]

    with open(yaml_file_path, "w") as f:
        f.write(f"path: {output_path}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: {num_classes}\n")
        f.write(f"names: {name_classes}\n")

if __name__ == "__main__":
    input_path = "data/football_train"
    output_detection_path = "data/football_dataset"
    output_classification_path = "data/football"

    train_val_split_ratio = 0.8
    create_classification_data = True
    create_detection_data = False

    if create_detection_data:
        process_detection_data(input_path, output_detection_path, train_val_split_ratio)

    if create_classification_data:
        process_classification_data(input_path, output_classification_path, train_val_split_ratio)

    print("Dataset creation completed")




         
         
   



