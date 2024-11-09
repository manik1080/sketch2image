import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image

# Hyperparams
image_size = 512  # check RAM capacity
batch_size = 4
num_epochs = 100
learning_rate = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Add more layers for larger image sizes
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            # Add more layers for larger image sizes
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, sketch, photo):
        x = torch.cat([sketch, photo], dim=1)
        return self.model(x)

# Custom Dataset for Paired Images
class SketchPhotoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.sketches_dir = os.path.join(root_dir, 'sketches')
        self.photos_dir = os.path.join(root_dir, 'photos')
        self.transform = transform
        self.sketches = sorted(os.listdir(self.sketches_dir))
        self.photos = sorted(os.listdir(self.photos_dir))

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketches_dir, self.sketches[idx])
        photo_path = os.path.join(self.photos_dir, self.photos[idx])

        sketch = Image.open(sketch_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")

        if self.transform:
            sketch = self.transform(sketch)
            photo = self.transform(photo)

        return sketch, photo

# Data Loading
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = SketchPhotoDataset(root_dir="train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and Optimizers
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training Loop
for epoch in range(num_epochs):
    for i, (sketches, photos) in enumerate(train_loader):
        sketches = sketches.to(device)
        photos = photos.to(device)
        real_labels = torch.ones(batch_size, 1, image_size // 16, image_size // 16).to(device)  # Downsampled label
        fake_labels = torch.zeros(batch_size, 1, image_size // 16, image_size // 16).to(device)

        # Train Generator
        optimizer_g.zero_grad()
        generated_photos = generator(sketches)
        gen_loss = criterion(discriminator(generated_photos, sketches), real_labels)
        gen_loss.backward()
        optimizer_g.step()

        # Train Discriminator
        optimizer_d.zero_grad()
        real_loss = criterion(discriminator(photos, sketches), real_labels)
        fake_loss = criterion(discriminator(generated_photos.detach(), sketches), fake_labels)
        dis_loss = (real_loss + fake_loss) / 2
        dis_loss.backward()
        optimizer_d.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                  f"D Loss: {dis_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")

    # Save sample outputs and model checkpoints
    if epoch % 10 == 0:
        save_image(generated_photos.data[:4], f"outputs/gen_epoch_{epoch}.png", normalize=True)
        torch.save(generator.state_dict(), f"models/generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch}.pth")
