# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Design and implement a Denoising Autoencoder using PyTorch to remove noise from handwritten digit images in the MNIST dataset. The model should take noisy images as input and learn to reconstruct the original clean images. Train the autoencoder using convolutional layers and evaluate its performance by visualizing the original, noisy, and denoised images after training.
<img width="1518" height="287" alt="image" src="https://github.com/user-attachments/assets/2ef75d92-9243-4c3b-ac95-a5ac07b26b85" />


## DESIGN STEPS

### Step 1: 
Import the required PyTorch, Torchvision, NumPy, and Matplotlib libraries. Configure the device to use GPU (CUDA) if available, otherwise use CPU.

### Step 2: 
Load the MNIST dataset and apply transformations to convert images into tensors. Create DataLoader objects for training and testing with a batch size of 128.

### Step 3: 
Define a function to add random noise to the input images. This simulates corrupted images which the model will learn to clean.

### Step 4: 
Build the Denoising Autoencoder model using convolutional layers for the encoder and transposed convolutional layers for the decoder to reconstruct the original image.

### Step 5: 
Initialize the model, loss function (MSELoss), and optimizer (Adam). Train the model for multiple epochs where noisy images are given as input and the original images are used as targets.

### Step 6: 
After training, visualize the results by displaying original images, noisy images, and denoised outputs to evaluate how well the model removes noise.


## PROGRAM
### Name:V KAMALESH VIJAYAKUMAR
### Register Number:212224110028

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),            
            nn.Conv2d(16, 8, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)              
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),    
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),    
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),             
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
summary(model, input_size=(1, 28, 28))

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
```

## OUTPUT

### Model Summary
<img width="1035" height="529" alt="image" src="https://github.com/user-attachments/assets/98fff247-4d75-45bd-84cb-064106d92fec" />


### Original vs Noisy Vs Reconstructed Image
<img width="1733" height="821" alt="image" src="https://github.com/user-attachments/assets/512a0bb8-ae22-45a0-9e7c-edd3665f7d5b" />


## RESULT

Thus to develop a convolutional autoencoder for image denoising application is done successfully.

