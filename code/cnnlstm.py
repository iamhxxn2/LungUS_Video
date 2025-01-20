import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet101, mobilenet_v3_large, mobilenet_v3_small

# CNNLSTM model
class CNNLSTM(nn.Module):
    def __init__(self, encoder, num_classes=4):
        super(CNNLSTM, self).__init__()
        
        self.cnn = encoder
        # case 3 : linear layer만 학습 진행
#         for param in self.cnn.features.parameters(): # Freeze the CNN except for the last layer
#             param.requires_grad = False

        # Replace the classifier with a new trainable layer
#         self.cnn.classifier = nn.Linear(self.cnn.classifier.in_features, 300)
        
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        # self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)
       
    def forward(self, x_in):
        hidden = None
        
        # ex: [4, 30, 3, 256, 256]
        batch_size, seq_len, channels, height, width = x_in.size()
        
        # Reshape to process each frame independently with CNN
        # ex: [120, 3, 256, 256]
        x_in = x_in.view(batch_size * seq_len, channels, height, width)  # Shape: [batch * 30, 3, 256, 256]
        
        # Apply CNN to each frame
        cnn_features = self.cnn(x_in)  # Shape: [batch * 30, 300]
        
        # Reshape to [batch, 30, 300] for LSTM input
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # Shape: [batch, 30, 300]
        
        # LSTM expects input of shape [batch, seq_len, input_size], which is [batch, 30, 300] here
        lstm_out, (hn, cn) = self.lstm(cnn_features)  # lstm_out shape: [batch, 30, 256]

        # Take the output from the last time step (out[-1]) for classification
        x = F.relu(lstm_out[:, -1, :])  # Shape: [batch, 256]
        
        # Fully connected layer for classification
        x = self.fc2(x)  # Shape: [batch, num_classes]

        return x
    
    '''
    # LSTM method 2 : Many-to-many
    def forward(self, x_in):
        batch_size, C, timesteps, H, W = x_in.size()
        lstm_outputs = torch.zeros(batch_size, timesteps, 256).to(x_in.device)

        # Process each timestep
        for t in range(timesteps):
            x_t = x_in[:, :, t, :, :]
            x_t = self.cnn(x_t)
            lstm_output, _ = self.lstm(x_t.unsqueeze(1))
            lstm_outputs[:, t, :] = lstm_output.squeeze(1)
        
        # Apply fc1 and relu to all time steps
        # x = self.fc1(lstm_outputs)
        x = F.relu(lstm_outputs)

        # Apply fc2 to all time steps
        x = self.fc2(x)

        # Take max across timesteps for each sample and each class
        # x, _ = torch.max(x, dim=1)
        return x
    '''
    
    