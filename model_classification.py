import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = self.make_block(in_channels=3, out_channels=8)
        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.conv4 = self.make_block(in_channels=32, out_channels=64)
        self.conv5 = self.make_block(in_channels=64, out_channels=64)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=3136, out_features=1024),
            nn.ReLU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def make_block(self, in_channels, out_channels, size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = out.view(x.shape[0], -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
    
if __name__ == "__main__":
    sample_data = torch.rand(16, 3, 224, 224)
    print(sample_data.shape)

    model = MyCNN(num_classes=12)
    output = model(sample_data)

    print(output.shape)
    print(output)

