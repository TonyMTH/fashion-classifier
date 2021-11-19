import preprocess as pr
import model as md
import torch
from torch import nn
from torch import optim

# Define Processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fetch Datasets
train_tranformer = pr.train_transform()
test_tranformer = pr.test_transform()

trainloader = pr.download_data('MNIST_data/', train_tranformer, True, 64)
testloader = pr.download_data('MNIST_data/', test_tranformer, False, 64)

# Fetch Model
model = md.model1
model.to(device)

# Parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 30
model_path = 'data/best_model.pt'
saved_model_device = torch.device("cpu")

# Train Model
pr.train_loop(model, epochs, trainloader, testloader, optimizer, criterion, model_path, saved_model_device, device)
