from dataset_classification import FootballDataset
from model_classification import MyCNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import ToTensor, Compose, Lambda

from tqdm.autonotebook import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
import os
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
        
def train():
    train_path = "data"
    val_path = "data"
    logging_path = "tensorboard"
    checkpoint_path = "trained_models"
    num_epochs = 50
    batch_size=64
    learning_rate = 1e-3
    momentum = 0.9
    num_classes = 12

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device found.")
    else:
        device = torch.device("cpu")
        print("CUDA device not found. Training on CPU.")    

    model = MyCNN(num_classes=num_classes).to(device)

    best_acc = -1
    start_epoch = 0

    # Check for last checkpoint
    last_checkpoint_path = os.path.join(checkpoint_path, "last.pt")
    metadata_path = os.path.join(checkpoint_path, "metadata.pt") 
    if resume and os.path.exists(last_checkpoint_path):
        print(f"Resume training from: {last_checkpoint_path}")
        model.load_state_dict(torch.load(last_checkpoint_path))
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            start_epoch = metadata["epoch"]
            best_acc = metadata["best_acc"]
        else:
            print("Error: metadata file not found.")
    else:
        print("Starting training from scratch.")

    transform = Compose([
        Lambda(resize_pad),
        ToTensor(), # Đưa kênh màu từ cuối lên đầu
    ])

    train_dataset = FootballDataset(root=train_path, train=True, transform=transform)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    val_dataset = FootballDataset(root=val_path, train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    num_iters = len(train_dataloader)

    if not os.path.isdir(logging_path):
        os.makedirs(logging_path)
    writer = SummaryWriter(logging_path)       

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    for epoch in range(start_epoch, num_epochs):
        # 1) Training stage
        model.train() # Chi cho mo hinh biet trong qua trong training, do co 1 so layers qua trinh train chay 1 kieu, qua trong val chay 1 kieu
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            # Forward
            images = images.to(device) # torch.Size([16, 3, 224, 224])
            labels = labels.to(device) # torch.Size([16])

            output = model(images)                 
            
            loss_value = criterion(output, labels)
            # print("Epoch {}/{}. Iter {}/{}. Loss {:0.4f}".format(epoch + 1, num_epochs, iter + 1, num_iters, loss_value))
            progress_bar.set_description(
                "Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, num_epochs, loss_value))
            writer.add_scalar(tag="Train/Loss", scalar_value=loss_value, global_step=(epoch * num_iters) + iter)

            # Backward
            optimizer.zero_grad() # xóa gradiant đã tích lũy từ các iteration trước, làm sạch buffer
            loss_value.backward() #chạy optimizer để tinh gradient cho toan bo tham so cua mo hinh
            optimizer.step() # update parameters cua mo hinh dựa vào gradient tính ở bước trên
        
        # 2) Validation stage
        model.eval() # 1 vài layers sẽ hoạt động khác đi so với quá trình train: dropout, batchnorm
        val_losses = []

        val_predictions = []
        val_labels = []

        progress_bar = tqdm(val_dataloader, colour="yellow")

        # with torch.inference_mode: # from pytorch 1.9
        with torch.no_grad():
            for iter, (images, labels) in enumerate(progress_bar):
                # print(epoch, images.shape, labels.shape)

                # Forward
                images = images.to(device) # torch.Size([16, 3, 224, 224])
                labels = labels.to(device) # torch.Size([16])

                output = model(images) 

                predictions = torch.argmax(output, dim=1)
                
                loss_value = criterion(output, labels) #loss_value là tensor kiểu scalar (0 chiều) = 1 value
                val_losses.append(loss_value.item())

                val_labels.extend(labels.cpu().tolist())
                val_predictions.extend(predictions.cpu().tolist())

        avg_loss = np.mean(val_losses)
        acc = accuracy_score(val_labels, val_predictions)
        print("Accuracy: {}. Average loss: {}".format(acc, avg_loss))

        writer.add_scalar(tag="Val/Loss", scalar_value=avg_loss, global_step=epoch)
        writer.add_scalar(tag="Val/Accuracy", scalar_value=acc, global_step=epoch)

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_path, "last.pt"))
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "best.pt"))
            best_acc = acc

        # Save metadata
        metadata = {
            "epoch": epoch + 1,
            "best_acc": best_acc
        }
        torch.save(metadata, metadata_path)
       
if __name__ == "__main__":
    resume = False
    train()