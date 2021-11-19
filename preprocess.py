from torchvision import datasets, transforms
import torch
import copy
import numpy as np


def download_data(path, transformer, datatype, batch_size):
    # Download and load the training data
    dataset = datasets.FashionMNIST(path, download=True, train=datatype, transform=transformer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_transform():
    # Define a transform to normalize the data
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5))
                               ])


def test_transform():
    # Define a transform to normalize the data
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5))
                               ])


def train_loop(model, epochs, trainloader, testloader, optimizer, criterion, model_path, saved_model_device, device):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    least_running_loss = np.inf

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Flatten Fashion-MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_loss = 0
        train_accuracy = 0
        test_accuracy = 0

        # Turn off gradients for validation, saves memory and computation
        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()

            # Validation pass
            for images, labels in testloader:
                # Move data to device
                images, labels = images.to(device), labels.to(device)

                images = images.view(images.shape[0], -1)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

            for images, labels in trainloader:
                # Move data to device
                images, labels = images.to(device), labels.to(device)

                images = images.view(images.shape[0], -1)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                train_accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train()
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Save best model
        if running_loss < least_running_loss:
            least_running_loss = running_loss

            best_model_state = copy.deepcopy(model)
            best_model_state.to(saved_model_device)
            torch.save(best_model_state, model_path)

        print("Epoch: {}/{}..".format(e + 1, epochs),
              "Training loss: {:.3f}..".format(running_loss / len(trainloader)),
              "Test loss: {:.3f}..".format(test_loss / len(testloader)),
              "Train Accuracy: {:.3f}".format(train_accuracy / len(trainloader)),
              "Test Accuracy: {:.3f}".format(test_accuracy / len(testloader))
              )
