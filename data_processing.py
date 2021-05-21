import cv2
import numpy as np
import os
from tqdm import tqdm

class MakeData():
	img_size = 256
	faces = "../Data/Faces"
	others = "../Data/Others"
	labels = {faces: 1, others: 0}

	processedData = []
	labelList = []
	faceCount = 0
	otherCount = 0

	def processData(self):
		for label in self.labels:
			print(label)
			for image in tqdm(os.listdir(label)):
				try:
					path = os.path.join(label, image)
					img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
					
					if label == self.faces:
						img = cv2.resize(img, (self.img_size, self.img_size))
					
					elif label == self.others:
						img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

					self.processedData.append(np.array(img))
					self.labelList.append(np.array(self.labels[label]))

					if label == self.faces:
						self.faceCount += 1

					elif label == self.others:
						self.otherCount += 1

				except Exception as e:
					pass

		self.processedData = np.array(self.processedData)
		self.labelList = np.array(self.labelList)
		np.save("imageData.npy", self.processedData)
		np.save("labels.npy", self.labelList)

		print("Faces: ", self.faceCount)
		print("Others: ", self.otherCount)

if __name__ == "__main__":
	process = MakeData()
	process.processData()