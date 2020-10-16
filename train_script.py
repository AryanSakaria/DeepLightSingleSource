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
params = {'batch_size': 6,
          'shuffle': False,
          'num_workers': 6}
max_epochs = 10

# # Datasets
# partition = # IDs
# labels = # Labels

# Generators
data_path = "/home/aryan/IIIT/4_1/IS/DeepLightSingleSource/Images/BlenderOutput"

training_set = DeepLightDataset(data_path)
training_generator = DataLoader(training_set, **params)

model = DeepLightModel()
model = model.to(device).double()

# validation_set = Dataset(partition['validation'], labels)
# validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for image, angle in training_generator:
        print(angle.shape, image.shape)
        # Transfer to GPU
        # print(local_batch.shape)

        # local_batch = local_batch.to(device)
        # out = model(local_batch.double())
        # print(out.shape)
        # Model computations
        

    # Validation
    # with torch.set_grad_enabled(False):
    #     for local_batch, local_labels in validation_generator:
    #         # Transfer to GPU
    #         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    #         # Model computations
    #         [...]