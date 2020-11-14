import os 
import numpy as np   
import torch 
from PIL import Image 
from torchvision import transforms as T
import math as m
# import cv2


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
	def __init__(self, root, transforms = None):
		shape = [os.path.join(root, file) for file in os.listdir(root)]
		self.images = [os.path.join(i, j) for i in shape for j in os.listdir(i)]
		self.root = root
		self.tsf = self.transform()
		self.angtsf = self.transform_angle()

	def transform(self):
		transforms = []
		transforms.append(T.ToTensor())
		transforms.append(T.Normalize((0.5, 0.5, 0.5,0.5), (0.5, 0.5, 0.5,0.5)))
		return T.Compose(transforms)

	def transform_angle(self):
		transforms = []
		transforms.append(T.ToTensor())
		return T.Compose(transforms)

	def cart2sph(self, x,y,z):
	    XsqPlusYsq = x**2 + y**2
	    r = m.sqrt(XsqPlusYsq + z**2)               # r
	    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
	    az = m.atan2(y,x)                           # phi
	    return r, elev, az

	def get_angle(self, file):
		f = open(file,"r")
		lines = f.readlines()
		c_vec = lines[0]
		c_vec = c_vec.strip().split('Camera')
		c_vec = c_vec[1].strip()
		c_vec = np.fromstring(c_vec, dtype=np.float32, sep=' ')
		c_r, c_elev, c_az = self.cart2sph(c_vec[0], c_vec[1],c_vec[2])
		l_vec = lines[1]
		l_vec = l_vec.strip().split('Light')
		l_vec = l_vec[1].strip()
		l_vec = np.fromstring(l_vec, dtype=np.float32, sep=' ')

		l_r, l_elev, l_az = self.cart2sph(l_vec[0], l_vec[1],l_vec[2])

		return [l_az - c_az, l_elev - c_elev]




	def __getitem__(self, idx):
		# print(self.images[idx])
		idx_path = self.images[idx]
		img_path = os.path.join(idx_path, "rgb.png")
		file_path = [depp_files for depp_files in os.listdir(idx_path)]
		for i in file_path:
			if "depth" in i:
				depth_path = os.path.join(idx_path, i)
				# print(i)
		ang_path = os.path.join(idx_path, "location.txt")

		im_cv = Image.open(img_path)
		im_cv = im_cv.convert("RGB")
		im_cv = np.array(im_cv, dtype=np.float32)
		im_cv = im_cv / 255

		im_d = Image.open(depth_path)
		im_d = im_d.convert("L")
		im_d = np.array(im_d, dtype=np.float32)
		im_d = im_d / np.max(im_d)
		im_ret = np.zeros((im_d.shape[0], im_d.shape[1], 4))
		im_ret[:,:,0:3] = im_cv
		im_ret[:,:, 3] = im_d

		ang = self.get_angle(ang_path)
		ang = np.array(ang).astype(dtype=np.double)
		ang = torch.from_numpy(ang)




		im_ret = self.tsf(im_ret)
		# ang_ret = self.angtsf(ang)
		# im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
		return im_ret, ang , idx_path


	def __len__(self):
		return len(self.images)


# path = "/home/aryan/IIIT/4_1/IS/DeepLightSingleSource/Images/BlenderOutput"
# dataset = DeepLightDataset(path)
#
# im, ang = dataset.__getitem__(0)