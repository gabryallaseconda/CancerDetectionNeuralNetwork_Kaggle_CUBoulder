
import os

import numpy as np

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim import Optimizer
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler

# AGGIUNGERE ANCHE IL CYCLING LEARNING RATE
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
#for epoch in range(10):
##data_loader = torch.utils.data.DataLoader(...)
#    for batch in data_loader:
#        train_batch(...)
#        scheduler.step()


import pretrainedmodels
import pretrainedmodels.utils as utils

import cv2

from skimage.io import imread

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Normalize, RandomGamma, RandomBrightnessContrast, HueSaturationValue, CLAHE, ChannelShuffle, 
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, PadIfNeeded, RandomCrop, Resize
)

def get_device():
    """
    Get a device object to run torch on accelerated hardware

    Returns:
        pytorch device: the most powerful device available
    """

    print("Setting DEVICE:")
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('\t MPS is available')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('\t CUDA is available')
    else:
        device = torch.device('cpu')
        print('\t No acceleration available')

    return device





def train(model, train_loader, optimizer, epoch, log_interval, loss_f, samples_per_epoch, scheduler):
    """Trains the model using the provided optimizer and loss function.
    Shows output each log_interval iterations 
    Args:
        model: Pytorch model to train.
        train_loader: Data loader.
        optimizer: pytroch optimizer.
        epoch: Current epoch.
        log_interval: Show model training progress each log_interval steps.
        loss_f: Loss function to optimize.
        samples_per_epoch: Number of samples per epoch to scale loss.
        cycling_optimizer: Indicates of optimizer is cycling.
    Returns:
        The mean loss across the epochs
    """
    # SET THE DEVICE
    device = get_device()

    # MODALITà DI ADDESTRAMENTO
    model.train()

    # TENERE TRACCIA DEL LOSS
    total_losses = []
    losses =[]

    # LOOP ON BATCHES, AS GIVEN BY THE LOADER
    for batch_idx, (x, target) in enumerate(train_loader):
        # AZZERA IL GRADIENTE NELL'OTTIMIZZATORE
        optimizer.zero_grad()
        
        output = model(x.to(device, dtype=torch.float)).squeeze(dim = 1)
        loss = loss_f(output, target.to(device, dtype=torch.float))
        losses.append(loss.item())
        loss.backward() # PASSO BACKWARD

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=4) # 4 IS THE MAX NORM. THE GRADIENT IS CLIPPED

        # SCHEDULER PER CYCLIC L R
        if scheduler:
            scheduler.step()

        
        # PLOTTING LOGS
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), samples_per_epoch,
                100. * batch_idx * len(x) / samples_per_epoch, np.mean(losses)))
            total_losses.append(np.mean(losses))
            losses = []

    train_loss_mean = np.mean(total_losses)
    print('Mean train loss on epoch {} : {}'.format(epoch, train_loss_mean))
    return train_loss_mean


def test(model, test_loader, loss_f):
    """Test the model with validation data.
    Args:
        model: Pytorch model to test data with.
        test_loader: Data loader.
        loss_f: Loss function.
    """
    
    # SET THE DEVICE
    device = get_device()

    # MODALITà DI VALUTAZIONE
    model.eval()

    # KEEP TRACK OF THE LOSS
    test_loss = 0 # THIS IS SOVRASCRITTO DOPO 
    predictions=[]
    targets=[]
    test_loss=[]

    # INFERENCE
    with torch.no_grad(): # DO NO TRACK THE GRADIENT DURING THE FOLLOWING OPERATIONS
        for x, target in test_loader: # QUI SI VA PER BATCH!
            output = model(x.to(device, dtype=torch.float)).squeeze(dim = 1)
            test_loss.append(loss_f(output, target.to(device, dtype=torch.float)).item())
            # RACCOLGO LE PREVIZIONI NELLA CPU           
            predictions.append(output.cpu())
            # RACCOLGO I VALORI REALI NELLA CPU
            targets.append(target.cpu())
    
    # IMPILO IN UN ARRAY UNIDIMENSIONALE
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)

    # CALCOLO LO SCORE
    score = roc_auc_score(targets, predictions)

    # PRINTO LA MEDIA DELLA PERDITA
    test_loss  = np.mean(test_loss)
    print('\nTest set: Average loss: {:.6f}, roc auc: {:.4f}\n'.format(test_loss, score))
    
    return test_loss, score	



class Net(nn.Module):
    """Build the nn network based on pretrained resnet models.
    Args:
        base_model: resnet34\resnet50\etc from pretrained models
        n_features: n features from last pooling layer       
    """
    def __init__(self, base_model, n_features):
        super(Net, self).__init__()
        # THESE GET THE LAYERS FROM THE BASE MODEL
        self.layer0 = nn.Sequential(*list(base_model.children())[:4])
        self.layer1 = nn.Sequential(*list(base_model.layer1))
        self.layer2 = nn.Sequential(*list(base_model.layer2))
        self.layer3 = nn.Sequential(*list(base_model.layer3))
        self.layer4 = nn.Sequential(*list(base_model.layer4))
        self.dense1 = nn.Sequential(nn.Linear(n_features, 128))
        self.dense2 = nn.Sequential(nn.Linear(128, 64))
        self.classif = nn.Sequential(nn.Linear(64, 1))

    # THIS USES THE BASE_MODEL
    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x 

    # THIS IS HOW TO CONDUCT THE FORWARD PASS
    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.classif(x)
        x = torch.sigmoid(x)
        return x

    

#####
## DATASET GENERATION, AUGMENTATION, LOADER
    

def aug_train(p=1): 
    return Compose([Resize(224, 224), 
                    HorizontalFlip(), 
                    VerticalFlip(), 
                    RandomRotate90(), 
                    OpticalDistortion(),
                    GridDistortion(border_mode=4), 
                    RandomBrightnessContrast(p=0.3), 
                    RandomGamma(p=0.3), 
                    OneOf([HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3), 
                           ChannelShuffle(p=0.3), 
                           CLAHE(p=0.3)])], p=p)

def aug_val(p=1):
    return Compose([
        Resize(224, 224)
    ], p=p)


class DataGenerator(data.Dataset):
    """Generates dataset for loading.
    Args:
        ids: images ids
        labels: labels of images (1/0)
        augment: image augmentation from albumentations
        imdir: path tpo folder with images
    """
    def __init__(self, ids, labels, augment, imdir):
        self.ids, self.labels = ids, labels
        self.augment = augment
        self.imdir = imdir
        
    def __len__(self):
        return len(self.ids) 

    def __getitem__(self, idx): # un singolo item!
        imid = self.ids[idx]
        y = self.labels[idx]
        X = self.__load_image(imid) # brutto nome, possiamo cambiarlo?

        #return X, np.expand_dims(y,0)
        
        # Modifica per bilanciare le classi, chiedete a chat gpt
        #if y == 0:
        #    num_augmentations = 15
        #else:
        #    num_augmentations = 10

        num_augmentations = 1 # LOOL

        augmented_images = [self.augment(image=X) for _ in range(num_augmentations)]
        augmented_images = [augmented['image'] / 255.0 for augmented in augmented_images]
        #augmented_images = [np.rollaxis(im, -1) for im in augmented_images]

        #return augmented_images, np.expand_dims(y, 0)

        augmented_images = [transforms.ToTensor()(im) for im in augmented_images][0] # LOOL
        return augmented_images, torch.tensor(y, dtype = torch.float32)


    def __load_image(self, imid): # brutto nome, possiamo cambiarlo?
        imid = imid+'.tif'
        im = imread(os.path.join(self.imdir, imid))
        
        # I seguenti commentati perchè spostati nella roba prima
        #augmented = self.augment(image=im)

        #im = augmented['image']
        
        # Min max normalization Normalization
        #im = im/255.0
        #im = np.rollaxis(im, -1)
        return im     
    

#######
### TEST TIME AUGMENTATION
    

def make_tta(image):
    '''
    return 4 pictures  - original, 3*90 rotations, mirror
    '''
    image_tta = np.zeros((4, image.shape[0], image.shape[1], 3))
    image_tta[0] = image
    aug = HorizontalFlip(p=1)
    image_aug = aug(image=image)['image']
    image_tta[1] = image_aug
    aug = VerticalFlip(p=1)
    image_aug = aug(image=image)['image']
    image_tta[2] = image_aug
    aug = Transpose(p=1)
    image_aug = aug(image=image)['image']
    image_tta[3] = image_aug    
    image_tta = np.rollaxis(image_tta, -1, 1)
    return image_tta


def aug_train_heavy(p=1):
    return Compose([HorizontalFlip(), 
                    VerticalFlip(), 
                    RandomRotate90(), 
                    Transpose(), 
                    RandomBrightnessContrast(p=0.3), 
                    RandomGamma(p=0.3), 
                    OneOf([
                        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3), 
                        ChannelShuffle(p=0.3)])], 
                    p=p)

heavy_tta = aug_train_heavy()

def make_tta_heavy(image, n_images=12):
    image_tta = np.zeros((n_images, image.shape[0], image.shape[1], 3))
    image_tta[0] = image/255.0
    for i in range(1,n_images):
        image_aug = heavy_tta(image=image)['image']
        image_tta[i] = image_aug/255.0
    image_tta = np.rollaxis(image_tta, -1, 1)
    return image_tta 

