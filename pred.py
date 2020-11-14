import torch
from torch.utils.data import DataLoader
from dataset.DeepLightDataset import DeepLightDataset
from model.DeepLightModel import DeepLightModel
import os
import re
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import numpy as np
import math as m
from sympy import *


def cart2sph(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    r = m.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(XsqPlusYsq))  # theta
    az = m.atan2(y, x)  # phi
    return r, elev, az


def get_angle(file):
    f = open(file, "r")
    lines = f.readlines()
    c_vec = lines[0]
    c_vec = c_vec.strip().split('Camera')
    c_vec = c_vec[1].strip()
    c_vec = np.fromstring(c_vec, dtype=np.float32, sep=' ')
    c_r, c_elev, c_az = cart2sph(c_vec[0], c_vec[1], c_vec[2])
    l_vec = lines[1]
    l_vec = l_vec.strip().split('Light')
    l_vec = l_vec[1].strip()
    l_vec = np.fromstring(l_vec, dtype=np.float32, sep=' ')

    l_r, l_elev, l_az = cart2sph(l_vec[0], l_vec[1], l_vec[2])
    return l_r, l_elev, l_az, c_r, c_elev, c_az

def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = rthetaphi[1] # to radian
    phi     = rthetaphi[2]
    x = r * sin( theta ) * cos( phi )
    y = r * sin( theta ) * sin( phi )
    z = r * cos( theta )
    return x,y,z

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

params = {'batch_size': 8,
          'shuffle': False,
          'num_workers': 8}

model = DeepLightModel()
model = model.to(device).double()
state_dict = torch.load("model_13.pth")

data_path = "images/new_data"

training_set = DeepLightDataset(data_path)
training_generator = DataLoader(training_set, **params)


new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

with torch.no_grad():
    for image, ang, path in training_generator:
        ang = ang.to(device)
        image = image.to(device)
        out = model(image.double())
        # print(out.shape)
        na = out.detach().to('cpu').numpy()
        # print(na.shape)
        for idx, i in enumerate(path):
            location_path = os.path.join(i,"location.txt")
            f = open(location_path, "r")
            lines = f.readlines()
            f.close()
            c_vec = lines[0]
            l_r, l_elev, l_az, c_r, c_elev, c_az = get_angle(location_path)
            # print(l_r, l_az, l_elev, na[idx])
            # print(c_r, c_az, c_elev)
            pred_az = na[idx][0] + c_az
            pred_elev = na[idx][1] + c_elev
            # print(asCartesian([l_r, pred_elev, pred_az]))
            x,y,z = asCartesian([l_r, pred_elev, pred_az])
            save_path = os.path.join(i,"pred.txt")
            l_vec = "Light " + str(x) + " " + str(y) + " " + str(z) + "\n"
            print(c_vec, l_vec)
            f = open(save_path, "w")
            f.write(c_vec+l_vec)
            f.close()


            # print(out[idx].shape)
            # print(location_path, os.path.exists(location_path))





