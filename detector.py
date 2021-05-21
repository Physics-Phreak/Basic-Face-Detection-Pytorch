import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("FaceDetector.pt")
model = model.to(device)
model.eval()

image_size = 256

def splitter(path):
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

	imgHight = img.shape[1]
	imgWidth = img.shape[0]
	
	resizeFactor = 1

	if imgHight >= 1280 or imgWidth >= 1280:
		resizeFactor = 0.7

	resizeHight = int(imgHight * resizeFactor)
	resizeWidth = int(imgWidth * resizeFactor)

	img = cv2.resize(img, (resizeHight, resizeWidth))

	splitImg = []
	imgData = img.tolist()

	splitDiff = 8
	numCols = len(imgData[0]) - image_size
	numRows = len(imgData) - image_size
	for col in tqdm(range(int(numCols/splitDiff))):
		for row in range(int(numRows/splitDiff)):
			split = []
			for section in imgData[(row*splitDiff): ((row*splitDiff) + image_size)]:
				split.append(section[(col*splitDiff):(image_size + (col*splitDiff))])

			splitImg.append(split)

	splitImg = np.array(splitImg, dtype="object")
	splitImg = splitImg.reshape(-1, image_size, image_size)
	return splitImg

def detector(splitImage):
	results = []
	bestResult = 0

	for image in tqdm(splitImage):
	
		image = torch.from_numpy(image.astype(np.float32))
		image = image.view(-1, 1, image_size, image_size)
		image = image.to(device)
		value = model(image)

		results.append(value.item())

	for result in results:
		if result > results[bestResult]:
			bestResult = results.index(result)

	return bestResult


if __name__ == '__main__':
	path = input("Path to image: ")

	plt.imshow((cv2.imread(path)))
	plt.show()

	data = splitter(path)
	data = data.astype(np.float)

	results = detector(data)

	plt.imshow(data[results])
	plt.show()