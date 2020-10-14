import os 
import numpy as np   


class DeepLightDataset:
	"""
	Dataloader for dataset
	3 channel RGB concattenated with 
	Depth image
	Output: 4 channel RGB-D and orientation
	"""
	"""
	TO-DO:
	Add orientation
	"""
	def __init__(self, data_dir, split="train"):
		# image_path = os.path.join(data_dir, split)
		image_path = data_dir
		images = [file for file in os.listdir(image_path)]
		# images = [img[0:-4] for img in images]

		self.images   = images
		self.data_dir = data_dir
		self.labels   = None

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img_idx = self.images[idx]
		

