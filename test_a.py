# Import necessary packages
import torch
import argparse
import torch.nn as nn
from torchmetrics.classification import Accuracy
from tqdm.auto import tqdm
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-t', '--data_path', default='DeepLearning4CV_challenge', help='path to the input train images')
ap.add_argument('-i', '--image_size', default=64, help='size of input image')
ap.add_argument('-b', '--batch_size', default=128, help='batch size')
args = vars(ap.parse_args())

# Setup data transforms
data_transforms = transforms.Compose([transforms.CenterCrop(args['image_size']),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load dataset
dataset = datasets.ImageFolder(args['data_path'], transform=data_transforms)
dataloader = DataLoader(dataset,
                        batch_size=args['batch_size'],
                        shuffle=True)

# Initialize class names
classes_names = dataset.classes

# split data
train_size = int(0.9 * len(dataloader))
val_size = len(dataloader) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataloader, [train_size, val_size])

# Create model
class AnimalsModel(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=16,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32x16
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)) # 16x16x32

        self.block_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)) # 8x8x64

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=output_shape),
        )

    def forward(self, x):
        return self.classifier(self.block_3(self.block_2(self.block_1(x))))


# initialize model, loss, accuracy, optimizer
model = AnimalsModel(3, len(classes_names)).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
acc_fn = Accuracy(task='multiclass', num_classes=len(classes_names)).to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)

# initialize train step
def train_step(model, train_dataset, optimizer, loss_fn, acc_fn, device):
    """Train model .

            Args:
                model : A PyTorch model capable of making predictions on train_dataset.
                train_dataset : The train dataset to train on.
                optimizer : Backward model.
                loss_fn : The loss function of model.
                acc_fn: An accuracy function to compare the models predictions to the truth labels.
                device: Send data to target device

            Returns:
                Model trained.
        """

    train_loss, train_acc = 0, 0
    for batch, (X_train, y_train) in enumerate(train_dataset.dataset):
        # Send X_train, y_train to GPU
        X_train, y_train = X_train.to(device), y_train.to(device)

        # 1.Forward
        y_pred = model(X_train)

        # 2.Compute loss, accuracy
        loss = loss_fn(y_pred, y_train)
        train_loss += loss
        train_acc += acc_fn(y_pred.argmax(axis=1), y_train)

        # 3.Optimizer zero grad
        optimizer.zero_grad()

        # 4.Loss backward
        loss.backward()

        # 5.Optimizer step
        optimizer.step()

    # Compute loss and accuracy per epoch and print
    train_loss /= len(train_dataset.dataset)
    train_acc /= len(train_dataset.dataset)
    print(f'Train loss: {train_loss:.5f} | Train accuracy: {train_acc * 100:.2f}%')


# Initialize test step
def test_step(model, test_dataset, loss_fn, acc_fn, device):
    """Test model .

                Args:
                    model : A PyTorch model capable of making predictions on test_dataset.
                    test_dataset : The target dataset to predict on.
                    loss_fn : The loss function of model.
                    acc_fn: An accuracy function to compare the models predictions to the truth labels.
                    device: Send data to target device
                Returns:
                    Metrics.
            """

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_val, y_val in test_dataset.dataset:
            # Send data to GPU
            X_val, y_val = X_val.to(device), y_val.to(device)

            # Compute predict
            y_pred_val = model(X_val)

            # Compute loss, accuracy
            test_loss += loss_fn(y_pred_val, y_val)
            test_acc += acc_fn(y_pred_val.argmax(axis=1), y_val)

        # Compute loss, accuracy per epoch and print
        test_loss /= len(test_dataset.dataset)
        test_acc /= len(test_dataset.dataset)
        print(f'Val loss: {test_loss:.5f}, Val accuracy: {test_acc * 100:.2f}%\n')


# Initialize epochs
epochs = 10

# Train model
for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch + 1}-----------')
    train_step(model, train_dataset, optimizer, loss_fn, acc_fn, device)
    test_step(model, val_dataset, loss_fn, acc_fn, device)

