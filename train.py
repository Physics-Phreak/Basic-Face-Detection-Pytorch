import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import FaceDetector
from data_processing import MakeData


if __name__ == "__main__":

	needData = False

	if needData:
		dataMaker = MakeData()
		dataMaker.processData()

		dataMaker = 0

	x_data = np.load("imageData.npy", allow_pickle=True)
	y_data = np.load("labels.npy", allow_pickle=True)
	y_data = y_data.reshape(y_data.shape[0], 1)

	xtrain, xtest, ytrain, ytest = train_test_split(x_data, y_data, test_size=0.2)

	xtrain = torch.from_numpy(xtrain.astype(np.float32))
	xtest = torch.from_numpy(xtest.astype(np.float32))
	ytrain = torch.from_numpy(ytrain.astype(np.float32))
	ytest = torch.from_numpy(ytest.astype(np.float32))

	trainDataset = TensorDataset(xtrain, ytrain)
	testDataset = TensorDataset(xtest, ytest)

	trainLoader = DataLoader(dataset=trainDataset,
							 batch_size=128,
							 shuffle=True,)

	testLoader = DataLoader(dataset=testDataset,
							batch_size=1,
							shuffle=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = FaceDetector().to(device)

	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters())

	scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

	epochs = 20

	best_acc = 0
	best_epoch = 0

	image_size = 256

	for epoch in range(epochs):
		for data in tqdm(trainLoader):
			
			optimizer.zero_grad()

			image = data[0].to(device)
			image = image.view(-1, 1, image_size, image_size)
			label = data[1].to(device)

			output = model(image)

			loss = criterion(output, label)
			loss.backward()

			optimizer.step()

		scheduler.step(loss.item())

		model.eval()

		correct = 0
		total = 0

		for image, label in tqdm(testLoader):
			image = image.to(device)
			image = image.view(-1, 1, image_size, image_size)
			label = label.to(device)

			output = model(image)

			if round(output.item()) == label.item():
				correct += 1

			total += 1

		acc = (correct/total)*100

		if acc > best_acc:
			best_acc = acc
			best_epoch = epoch + 1
			torch.save(model, "FaceDetector.pt")

		model.train()

		print(f"epoch: {epoch + 1}, loss: {loss.item()}, accuracy: {acc}")

	print(f"best epoch: {best_epoch}, best accuracy: {best_acc}")