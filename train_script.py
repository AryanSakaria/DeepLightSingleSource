import torch
from torch.utils.data import DataLoader
from dataset.DeepLightDataset import DeepLightDataset
from model.DeepLightModel import DeepLightModel


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
# torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 8,
          'shuffle': False,
          'num_workers': 12}
max_epochs = 10

# Generators
data_path = "images/500 Cubes 160_120/BlenderOutput"

training_set = DeepLightDataset(data_path)
training_generator = DataLoader(training_set, **params)

model = DeepLightModel()
model = model.to(device).double()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# validation_set = Dataset(partition['validation'], labels)
# validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# Loop over epochs
criteria = torch.nn.MSELoss()
for epoch in range(max_epochs):
    # Training
    for image, ang in training_generator:
        optimizer.zero_grad()
        # print(image.shape)
        # print(ang.shape)
        # Transfer to GPU
        # print(local_batch.shape)

        # local_batch = local_batch.to(device)
        ang = ang.to(device)
        image = image.to(device)
        out = model(image.double())
        loss = criteria(out, ang)
        print(loss.item)
        loss.backward()
        optimizer.step()
        # print(out.shape)
        # Model computations