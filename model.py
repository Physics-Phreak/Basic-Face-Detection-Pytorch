import torch
import torch.nn as nn

class FaceDetector(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 16, 5) #output dim = ((input dim - kernel size + 2*padding dim)/stride) + 1 = 256 - 5 + 1 = 252
		self.conv2 = nn.Conv2d(16, 8, 3) #output dim = 126(after pooling) - 3 + 1 = 124
		self.conv3 = nn.Conv2d(8, 3, 3) #output dim = 62 - 3 + 1 = 60
		self.pool = nn.MaxPool2d(2, stride=2)
		self.fc1 = nn.Linear(3 * 30 * 30, 128)
		self.fc2 = nn.Linear(128, 1)
		self.relu = nn.ReLU()
		self.dropout1 = nn.Dropout(0.8)
		self.dropout2 = nn.Dropout(0.5)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.pool(x)

		x = self.relu(self.conv2(x))
		x = self.pool(x)
		
		x = self.relu(self.conv3(x))
		x = self.pool(x)

		x = x.view(-1, 3 * 30 * 30)

		x = self.dropout2(self.relu(self.fc1(x)))
		x = self.dropout1(self.fc2(x))

		return self.sigmoid(x)