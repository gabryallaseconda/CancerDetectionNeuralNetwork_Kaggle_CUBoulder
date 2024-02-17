# Python stl
import os

# Data tools
import numpy as np
from sklearn.metrics import roc_auc_score

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils import data

# Image manipulation tools
from skimage.io import imread
from albumentations import (
    HorizontalFlip, 
    VerticalFlip, 
    IAAPerspective, 
    ShiftScaleRotate, 
    CLAHE, 
    RandomRotate90, 
    Normalize, 
    RandomGamma, 
    RandomBrightnessContrast, 
    HueSaturationValue, 
    ChannelShuffle, 
    Transpose, 
    ShiftScaleRotate, 
    Blur, 
    OpticalDistortion, 
    GridDistortion, 
    HueSaturationValue,
    IAAAdditiveGaussianNoise, 
    GaussNoise, 
    MotionBlur, 
    MedianBlur, 
    IAAPiecewiseAffine,
    IAASharpen, 
    IAAEmboss, 
    RandomContrast, 
    RandomBrightness, 
    Flip, 
    OneOf, 
    Compose, 
    PadIfNeeded, 
    RandomCrop, 
    Resize
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





def train(model, train_loader, optimizer, epoch, log_interval, loss_f, scheduler, device, samples_per_epoch = None):
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

        optimizer.step()

        # SCHEDULER PER CYCLIC L R
        if scheduler:
            scheduler.step()

        
        # PLOTTING LOGS
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}] \tLoss: {:.6f}'.format(
                        epoch, 
                        batch_idx * len(x), 
                        np.mean(losses))
                        )
            total_losses.append(np.mean(losses))
            losses = []

    train_loss_mean = np.mean(total_losses)
    print('Mean train loss on epoch {} : {}'.format(epoch, train_loss_mean))
    return train_loss_mean


def test(model, test_loader, loss_f, device):
    """Test the model with validation data.
    Args:
        model: Pytorch model to test data with.
        test_loader: Data loader.
        loss_f: Loss function.
    """
    


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
    
    
    # Here we have a problem. We cannot use vstack because batches has different dimension.
    # We should check the cause for this and look for a solution 

    #predictions = np.vstack(predictions)
    #targets = np.vstack(targets)

    predictions_flattened = []
    for pr in predictions:
        predictions_flattened.extend(pr)

    targets_flattened = []
    for tar in targets:
        targets_flattened.extend(tar)

    # CALCOLO LO SCORE
    score = roc_auc_score(targets_flattened, predictions_flattened)

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
                           CLAHE(p=0.3)])
                    ], p=p)

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

    def __getitem__(self, idx): 
        imid = self.ids[idx]
        y = self.labels[idx]
        X = self.load_image(imid) 

        # Different augmentations strategies depending on the label
        # This is the strategy to balance the classes
        if y == 0:
            num_augmentations = 15
        else:
            num_augmentations = 10

        augmented_images = [self.augment(image=X) for _ in range(num_augmentations)]
        augmented_images = [augmented['image'] / 255.0 for augmented in augmented_images]

        augmented_images = [transforms.ToTensor()(im) for im in augmented_images][0] # LOOL
        return augmented_images, torch.tensor(y, dtype = torch.float32)


    def load_image(self, imid): 
        imid = imid+'.tif'
        im = imread(os.path.join(self.imdir, imid))
        
        return im     
    


class DataGeneratorInference(data.Dataset):
    """Generates dataset for loading inference data.
    Args:
        ids: images ids
        labels: labels of images (1/0)
        augment: image augmentation from albumentations
        imdir: path tpo folder with images
    """
    def __init__(self, ids, imdir):
        self.ids = ids
        self.imdir = imdir
        
    def __len__(self):
        return len(self.ids) 

    def __getitem__(self, idx): 
        
        imid = self.ids[idx]
        imid = imid+'.tif'
        image = imread(os.path.join(self.imdir, imid))

        # Format manipulation
        image = transforms.ToTensor()(image)
        image =  torch.tensor(image, dtype = torch.float32)

        return image



def produce_test_time_augmentation(image, number):

    # Set the augmentation
    resize_only = Compose([Resize(224,224)], p=1)

    augmentation = Compose([Resize(224, 224), 
                            HorizontalFlip(p=0.5),  # Perchè non c'è il resize????
                            VerticalFlip(p=0.5), 
                            RandomRotate90(), 
                            Transpose(),   # Che cos'è?
                            RandomBrightnessContrast(p=0.3), 
                            RandomGamma(p=0.3), 
                            OneOf([     HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3), 
                                        ChannelShuffle(p=0.3), 
                                        CLAHE(p=0.3)])
                            ], p=1)

    # Initialize images vector
    images = np.zeros((number, image.shape[0], image.shape[1], image.shape[2]))#3))

    # First image (original)
    images[0] = resize_only(image = image)['image']/255.0

    # Others
    for i in range(1, number):
        images[i] = augmentation(image = image)['image'] / 255.0

    # Switch axis
    return np.moveaxis(images, -1, 1)

