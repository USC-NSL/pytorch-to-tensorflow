from simple_model import SimpleModel

import numpy as np
import torch

test_size = 2000

input_size = 20
hidden_sizes = [50, 50]
output_size = 1

X_test = np.random.randn(test_size, input_size).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_pytorch = SimpleModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
model_pytorch = model_pytorch.to(device) # important...

model_pytorch.load_state_dict(torch.load('./models/model_simple.pt'))

dummy_input = torch.from_numpy(X_test[0].reshape(1, -1)).float().to(device)
dummy_output = model_pytorch(dummy_input)
print(dummy_output)

# Export to ONNX format
torch.onnx.export(model_pytorch, dummy_input, './models/model_simple.onnx', input_names=['test_input'], output_names=['test_output'])
