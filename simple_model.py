import torch.nn as nn

# Build model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fcs = []  # List of fully connected layers
        in_size = input_size
        
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_features=in_size, out_features=next_size)
            in_size = next_size
            self.__setattr__('fc{}'.format(i), fc)  # set name for each fullly connected layer
            self.fcs.append(fc)
            
        self.last_fc = nn.Linear(in_features=in_size, out_features=output_size)
        
    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            x = nn.ReLU()(x)
        out = self.last_fc(x)
        return nn.Sigmoid()(out)
