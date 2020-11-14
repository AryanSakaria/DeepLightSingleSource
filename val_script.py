import torch
from torch.utils.data import DataLoader
from dataset.DeepLightDataset import DeepLightDataset
from model.DeepLightModel import DeepLightModel
import os
import re
from tqdm import tqdm




# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
# torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 30,
          'shuffle': False,
          'num_workers': 30}
max_epochs = 10

# Generators
data_path = "images/500 Cubes 160_120/BlenderOutput"

training_set = DeepLightDataset(data_path)
training_generator = DataLoader(training_set, **params)
model = DeepLightModel()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# validation_set = Dataset(partition['validation'], labels)
# validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# Loop over epochs
# if not os.path.isdir("model_weights"):
#     os.mkdir("model_weights")
#     start_epoch = 0
# else:
#     save_path = newest("model_weights")
#     start_epoch = int(re.findall(r'\d+', save_path)[0])
#     model.load_state_dict(torch.load("model_weights/model_" + str(start_epoch) + '.pth'))

model.load_state_dict(torch.load("model_weights/model_9.pth"))
model = torch.nn.DataParallel(model)
model = model.to(device).double()

criteria = torch.nn.MSELoss()
# for epoch in range(start_epoch, max_epochs):
#     # Training
idx = 0
running_loss = 0
with torch.no_grad():
    for image, ang in tqdm(training_generator):
        idx += 1
        ang = ang.to(device)
        image = image.to(device)
        out = model(image.double())
        loss = criteria(out, ang)
        running_loss += loss.item() * 30

print(running_loss/idx)
