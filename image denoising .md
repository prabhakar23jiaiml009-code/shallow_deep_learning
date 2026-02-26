import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using Device:", device)

train_path = "/Users/yourname/Desktop/fashion_dataset/train"
test_path  = "/Users/yourname/Desktop/fashion_dataset/test"

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.Grayscale(num_output_channels=3),  # convert to 3 channel
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
test_dataset  = datasets.ImageFolder(root=test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

def add_noise(img):
    noise = torch.randn_like(img) * 0.2
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)

class StackedAutoencoder(nn.Module):
    def __init__(self):
        super(StackedAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = StackedAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        noisy_images = add_noise(images)

        outputs = model(noisy_images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training Complete!")

dataiter = iter(test_loader)
images, _ = next(dataiter)
images = images.to(device)
noisy_images = add_noise(images)
outputs = model(noisy_images)

def imshow(img):
    img = img.cpu().detach().numpy()
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.axis("off")
    plt.show()

print("Original Image")
imshow(images[0])

print("Noisy Image")
imshow(noisy_images[0])

print("Denoised Image")
imshow(outputs[0])
