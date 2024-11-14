import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import ast


class DeepFashion2SegmentationDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Extract data for the current item
        img_path = self.data.loc[idx, 'path']
        img_full_path = os.path.join(self.image_dir, os.path.basename(img_path))
        segmentation = self.data.loc[idx, 'segmentation']
        
        # Load image
        image = cv2.imread(img_full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a blank mask with the same dimensions as the image
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Parse the segmentation string into a list of polygons
        polygons = ast.literal_eval(segmentation)
        
        # Fill each polygon on the mask
        for poly in polygons:
            points = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [points], color=1)
        
        # Transform the image and mask to tensors if specified
        if self.transform:
            image = self.transform(image)
            mask = ToTensor()(mask).float()
        
        return image, mask


# Display a sample image and mask for verification
def display_sample(dataset, index):
    image, mask = dataset[index]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image.permute(1, 2, 0).numpy())
    ax1.set_title('Original Image')
    ax2.imshow(mask.squeeze().numpy(), cmap='gray')
    ax2.set_title('Segmentation Mask')
    plt.show()






class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output layer for single channel mask
        )

    def forward(self, x):
        # Encode
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        # Decode
        x = self.upconv1(x)
        x = self.dec1(x)
        
        # Output activation with sigmoid for binary segmentation
        return torch.sigmoid(x)




def training(model, optimizer):
    # Set number of epochs
    num_epochs = 1

    # Training loop
    for epoch in range(num_epochs):
        for images, masks in dataloader:
            # Zero out gradients from previous batch
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


def visulaize():
    # Display a batch of original images, ground truth masks, and predicted masks
    for images, masks in dataloader:
        outputs = model(images)  # Get predictions
        
        for i in range(len(images)):
            plt.figure(figsize=(15, 5))
            
            # Original Image
            plt.subplot(1, 3, 1)
            plt.imshow(images[i].permute(1, 2, 0).numpy())
            plt.title("Original Image")

            # Ground Truth Mask
            plt.subplot(1, 3, 2)
            plt.imshow(masks[i].squeeze().numpy(), cmap='gray')
            plt.title("Ground Truth Mask")

            # Predicted Mask
            plt.subplot(1, 3, 3)
            plt.imshow(outputs[i].detach().squeeze().numpy(), cmap='gray')
            plt.title("Predicted Mask")

            plt.show()
        break  # Display only one batch




if __name__ == "__main__":
    # Directory paths
    csv_file = 'C:/Users/bhara/Desktop/clothfilter/Dataset/DeepFashion2/input/updatedTrain.csv'  # Update as per your CSV file name
    image_dir = 'C:/Users/bhara/Desktop/clothfilter/Dataset/DeepFashion2/resized/train'

    # Load dataset and display sample data
    transform = ToTensor()
    dataset = DeepFashion2SegmentationDataset(csv_file=csv_file, image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Show a sample
    display_sample(dataset, 0)


    # Initialize model
    model = UNet()


    # Define binary cross-entropy loss for binary segmentation
    criterion = nn.BCELoss()
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    training(model, optimizer)
    visulaize(dataloader, model)