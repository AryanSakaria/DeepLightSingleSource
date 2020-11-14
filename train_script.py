import torch
from torch.utils.data import DataLoader
from dataset.DeepLightDataset import DeepLightDataset
from model.DeepLightModel import DeepLightModel
import os
import re
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
device_cpu = torch.device("cpu")
model = DeepLightModel()
model = model.to(device_cpu).double()
print(device)
# torch.backends.cudnn.benchmark = True
writer = SummaryWriter('plots_loss/')

# Parameters
params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 16}
max_epochs = 10

# Generators
data_path = "images/new_data"

training_set = DeepLightDataset(data_path)
training_generator = DataLoader(training_set, **params)

dataiter = iter(training_generator)
images, angels, path_idx = dataiter.next()

writer.add_graph(model, images)
writer.close()

params = {'batch_size': 32,
          'shuffle': False,
          'num_workers': 16}

training_set = DeepLightDataset(data_path)
training_generator = DataLoader(training_set, **params)
model = model.to(device).double()

print("Len dataloader ")
print(len(training_generator))
# validation_set = Dataset(partition['validation'], labels)
# validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# Loop over epochs
if not os.path.isdir("model_weights2"):
    os.mkdir("model_weights2")
    start_epoch = 0
else:
    save_path = newest("model_weights2")
    start_epoch = int(re.findall(r'\d+', save_path)[0])
    model.load_state_dict(torch.load("model_weights2/model_" + str(start_epoch) + '.pth'))
    start_epoch += 1

model = torch.nn.DataParallel(model)
model = model.to(device).double()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
print("start_epoch", start_epoch)

criteria = torch.nn.MSELoss()
for epoch in range(start_epoch, 20):
    # Training
    running_loss = 0.0
    i = 0
    for image, ang, path in tqdm(training_generator):
        i += 1

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
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 50 == 49:  # every 1000 mini-batches...
            # ...log the running loss
            writer.add_scalar('training loss',
                              running_loss / 50,
                              epoch * len(training_generator) + i)

    # print("epoch loss :", running_loss/)
    if epoch % 1 == 0:
        newSavePath = 'model_weights2/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), newSavePath)














