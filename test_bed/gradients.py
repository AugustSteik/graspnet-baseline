import torch
import torch.nn as nn

# Define simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.fc = nn.Linear(1, 1)  # Fully connected layer

    def forward(self, x):
        x = self.conv(x)  # Convolution
        x = x.view(x.shape[0], -1)
        x = self.fc(x)  # Fully connected layer
        return x

# Create model
model = SimpleCNN()

# Define input
x = torch.ones(1, 1, 3, 3, requires_grad=True)  # Batch=1, Channels=1, 3x3 image

# Forward pass
output = model(x)

# Backward pass
output.backward()

# Print gradients
print("Gradient of input w.r.t. output:\n", x.grad)