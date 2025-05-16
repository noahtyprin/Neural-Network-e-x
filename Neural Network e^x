import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Load data from CSV file
data = pd.read_csv('e^x Datapoints.csv', sep='\t')
inputs = data['Inputs'].values.reshape(-1, 1)
targets = data['Outputs'].values.reshape(-1, 1)

# Convert to PyTorch tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

# Normalize inputs to [0, 1] range
input_max = inputs.max()
inputs = inputs / input_max

# Create results file if it doesn't exist
if not os.path.exists('e^2_predictions.csv'):
    pd.DataFrame(columns=['Epochs', 'Predicted', 'Actual']).to_csv('e^2_predictions.csv', index=False)

# 2. Define the 2-layer neural network
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()  # Changed to ReLU
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

# 3. Instantiate the model
model = TwoLayerNet(input_size=1, hidden_size=20, output_size=1)  

# 4. Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  

# 5. Train the model
epochs = 100000
while True:
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # Add gradient clipping with smaller max_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        if (epoch+1) % 10000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # 6. Test the model
    testVal = float(input("Enter a value to test e^x: "))
    testX = torch.tensor([[testVal / input_max]], dtype=torch.float32)  # Normalize test input
    predicted = model(testX)
    print(f'Predicted e^{testVal} = {predicted.item()}, Actual = {np.exp(testVal)}')
    
    # Save e^2 prediction to CSV
    test2 = torch.tensor([[2 / input_max]], dtype=torch.float32)
    pred2 = model(test2)
    results = pd.DataFrame({
        'Epochs': [epochs],
        'Predicted': [pred2.item()],
        'Actual': [np.exp(2)]
    })
    results.to_csv('e^2_predictions.csv', mode='a', header=False, index=False)
    
    retrain = input("Would you like to run the network for another 100000 epochs? (y/n): ")
    if retrain.lower() != 'y':
        break
