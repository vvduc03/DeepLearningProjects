# Import necessary packages
import torch
import cv2 as cv
import argparse
import torchvision
import tensorboard
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import requests
import torch.nn as nn
from tqdm.auto import tqdm
from torchmetrics.classification import Accuracy
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-t', '--data_path', default='data/Animals', help='path to the input train images')
ap.add_argument('-e', '--epochs', default=3, help='numbers of epochs to train model')
ap.add_argument('-i', '--image_size', default=224, help='size of input image')
ap.add_argument('-b', '--batch_size', default=64, help='batch size')
ap.add_argument('-g', '--input_channels', default=3, help='number of input channels')
ap.add_argument('-p', '--patch_size', default=16, help='size of each patch image')
ap.add_argument('-a', '--embedding_dims', default=768, help='size of embedding')
args = vars(ap.parse_args())

# Setup data transforms
data_transforms = transforms.Compose([transforms.Resize(args['image_size']),
                                      transforms.CenterCrop(args['image_size']),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
# Load dataset
data_set = datasets.ImageFolder(args['data_path'], transform=data_transforms)

# split data
train_val_size = int(0.9 * len(data_set))
test_size = len(data_set) - train_val_size
train_val_dataset, test_dataset = torch.utils.data.random_split(data_set, [train_val_size, test_size])

train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset,
                              batch_size=args['batch_size'],
                              shuffle=True)
val_dataloader = DataLoader(val_dataset,
                            batch_size=args['batch_size'],
                            shuffle=False)
test_dataloader = DataLoader(test_dataset,
                             batch_size=args['batch_size'],
                             shuffle=False)
classes_names = data_set.classes

# Initialize targets test list
test_targets = []
for i in range(len(test_dataset)):
    _, target = test_dataset[i]
    test_targets.append(target)
test_targets = torch.Tensor(test_targets)

# Get the length of class_names (one output unit for each class)
output_shape = len(classes_names)

# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turn  a 2D image to 1D learnable embedding vector

    Args:
        in_channels (int): Numbers of channels for input images. Defaults to 3
        patch_size (int): Size of each patch image. Defaults to 16
        embedding_dims (int): Size embedding to turn image into. Defaults to 768
    """

    # 2.Initialize class store variables
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dims:int=768):
        super().__init__()

        self.patch_size = patch_size
        # 3.Create layers turn image to patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                out_channels=embedding_dims,
                                kernel_size=patch_size,
                                stride=patch_size,
                                padding=0)

        # 4.Create layer flatten patches
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)
    # 5.Define forward method
    def forward(self, x):
        # Check resolution of image need divisble by patch size
        img_resolution = x.shape[-1]
        assert img_resolution % self.patch_size == 0, f'Input image size must divisble by patch size, image size: {img_resolution}, patch size: {self.patch_size}'

        # 6.Performers forward
        x_patcher = self.patcher(x)
        x_flatten = self.flatten(x_patcher)

        # Make sure right order output image
        return x_flatten.permute(0, 2, 1)


# 1. Create a class that inherits from nn.Module
class MSABlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """

    # 2. Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 attn_dropout: float = 0):  # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)  # does our batch dimension come first?

    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,  # query embeddings
                                             key=x,  # key embeddings
                                             value=x,  # value embeddings
                                             need_weights=False)  # do we need the weights or just the layer outputs?
        return attn_output


# 1. Create a class that inherits from nn.Module
class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""

    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim: int = 768,  # Hidden Size D from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 dropout: float = 0.1):  # Dropout from Table 3 for ViT-Base
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),  # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,  # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim),  # take back to embedding_dim
            nn.Dropout(p=dropout)  # "Dropout, when used, is applied after every dense layer.."
        )

    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

# 1.Create Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    """Initialize Transformer Encoder Block"""

    # 2.Store hyperparameters
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 attn_dropout: int = 0):
        super().__init__()

        # 3.Initialize MSA Block
        self.MSA = MSABlock(embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            attn_dropout=attn_dropout)

        # 4.Initialize MLP Block
        self.MLP = MLPBlock(embedding_dim=embedding_dim,
                            mlp_size=mlp_size,
                            dropout=mlp_dropout)

    # 5.Create forward method
    def forward(self, x):

        # 6.Initialize residual connection for MSA Block
        x = self.MSA(x) + x

        # 7.Initialize residual connection for MLP Block
        x = self.MLP(x) + x

        return x

class PositionTokenEmbedding(nn.Module):
    """Token and position embedding"""
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 embedding_dims: int = 768,
                 embedding_dropout: float = 0.1,
                 batch_size: int = 32):
        super().__init__()

        # Token class
        self.token_class = nn.Parameter(torch.randn(1, 1, embedding_dims),
                                        requires_grad=True)

        # Position
        self.num_patches = int((img_size * img_size) / (patch_size ** 2))
        self.position = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dims),
                                     requires_grad=True)

        # Embedding dropout
        self.dropout = nn.Dropout(p=embedding_dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.token_class.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = self.position + x
        x = self.dropout(x)

        return x

# 1.Create ViT class
class ViT(nn.Module):
    """Create a Vision Transformer architecture with ViT-Base hyperparameters"""
    def __init__(self,
                 img_size: int = 224,  # Training resolution from Table 3 in ViT paper
                 in_channels: int = 3,  # Number of channels in input image
                 batch_size: int = 64,  # Batch size
                 patch_size: int = 16,  # Patch size
                 num_transformer_layers: int = 12,  # Layers from Table 1 for ViT-Base
                 embedding_dims: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 attn_dropout: float = 0,  # Dropout for attention projection
                 mlp_dropout: float = 0.1,  # Dropout for dense/MLP layers
                 embedding_dropout: float = 0.1,  # Dropout for patch and position embeddings
                 num_classes: int = 1000):  # Default for ImageNet ):
        super().__init__()

        # 2.Initialize block
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dims=embedding_dims)

        self.position_class = PositionTokenEmbedding(img_size=img_size,
                                                    patch_size=patch_size,
                                                    embedding_dims=embedding_dims,
                                                    embedding_dropout=embedding_dropout,
                                                    batch_size=batch_size)

        self.transformers_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dims,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout,
                                                                            attn_dropout=attn_dropout)
                                                    for _ in range(num_transformer_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dims),
            nn.Linear(in_features=embedding_dims,
                      out_features=num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.position_class(x)
        x = self.transformers_encoder(x)
        x = self.classifier(x[:, 0])

        return x

# Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)
vit = ViT(num_classes=len(classes_names)).to(device)

optimizer = torch.optim.Adam(params=vit.parameters(),
                             lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                             betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                             weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

# Setup the loss function for multi-class classification
loss_fn = torch.nn.CrossEntropyLoss()
acc_fn = Accuracy(task='multiclass', num_classes=len(classes_names)).to(device)
# initialize train step
def train_step(model, train_dataloader, optimizer, loss_fn, acc_fn, device):
    """Train model .

            Args:
                model : A PyTorch model capable of making predictions on train_dataloader.
                train_dataloader : The train dataset to train on.
                optimizer : Backward model.
                loss_fn : The loss function of model.
                acc_fn: An accuracy function to compare the models predictions to the truth labels.
                device: Send data to target device

            Returns:
                Model trained.
        """

    train_loss, train_acc = 0, 0
    for batch, (X_train, y_train) in enumerate(train_dataloader):
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

        # Print out how many samples have been seen
        if batch % 10 == 0:
            print(f"Looked at {batch * len(X_train)}/{len(train_dataloader.dataset)} samples")

    # Compute loss and accuracy per epoch and print
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f'Train loss: {train_loss:.5f} | Train accuracy: {train_acc * 100:.2f}%')
    return train_loss, train_acc

# Initialize val step
def val_step(model, val_dataloader, loss_fn, acc_fn, device):
    """Test model .

                Args:
                    model : A PyTorch model capable of making predictions on test_dataloader.
                    val_dataloader : The target dataset to predict on.
                    loss_fn : The loss function of model.
                    acc_fn: An accuracy function to compare the models predictions to the truth labels.
                    device: Send data to target device
                Returns:
                    Metrics.
            """

    val_loss, val_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_val, y_val in val_dataloader:
            # Send data to GPU
            X_val, y_val = X_val.to(device), y_val.to(device)

            # Make predictions with the model
            y_pred_val = model(X_val)

            # Compute loss, accuracy
            val_loss += loss_fn(y_pred_val, y_val)
            val_acc += acc_fn(y_pred_val.argmax(axis=1), y_val)

        # Compute loss, accuracy per epoch and print
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
        print(f'Val loss: {val_loss:.5f} | Val accuracy: {val_acc * 100:.2f}%\n')
        return val_loss, val_acc

# Initialize evaluate model step
def eval_model(model, test_dataloader, loss_fn, acc_fn):
    """Last test model.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        test_dataloader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        acc_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        Metrics.
    """

    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions"):
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # Make predictions with the model
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += acc_fn(y_pred.argmax(axis=1), y)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(test_dataloader)
        acc /= len(test_dataloader)

    return loss, acc

for epoch in tqdm(range(args['epochs']), desc='Training model'):
    print(f'Epoch: {epoch + 1}-----------')
    train_loss, train_acc = train_step(vit, train_dataloader, optimizer, loss_fn, acc_fn, device)
    val_loss, val_acc = val_step(vit, val_dataloader, loss_fn, acc_fn, device)

# Evaluate model
loss, acc = eval_model(vit, test_dataloader, loss_fn, acc_fn)
print(f'Test loss: {loss:.5f} | Test accuracy: {acc * 100:.2f}%\n')