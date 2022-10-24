#!/usr/bin/env python
# coding: utf-8

# Data from: http://zhao-nas.bio.cmu.edu:5000/fsdownload/aBDx29J7H/Ensemble%20learning%20data_shared

# In[1]:


# !pip install wandb
# !pip install pytorch_lightning
from pathlib import Path
import cv2 , os, numpy as np, torch, pandas as pd, tqdm as tqdm, PIL.Image as Image, time, IPython
#from pylab import rcParams
import datetime
# 
from torch import nn, optim
import torch.nn.functional as F
#import torchvision.transforms as T
from pytorch_lightning.plugins import DDPPlugin

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchsummary import summary
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from albumentations.pytorch.transforms import ToTensorV2 
from pytorch_lightning.callbacks import  GradientAccumulationScheduler
from torch.optim import lr_scheduler
#
from numpy.lib.function_base import select
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#
import pytorch_lightning as pl, torchmetrics
import albumentations as A

import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"  # nccl (not for windows)
# https://pytorch.org/docs/stable/distributed.html
#     Rule of thumb
#         Use the NCCL backend for distributed GPU training
#         Use the Gloo backend for distributed CPU training

os.environ["CUDA_LAUNCH_BLOCKING"]="1"
AVAIL_GPUS = torch.cuda.device_count()
AVAIL_CPUS = os.cpu_count()
print(f"GPUS:{AVAIL_GPUS}|CPUS:{AVAIL_CPUS}")

class PATH_ARGS:
    proj_path = Path('./').absolute()  # [CHANGE THIS for new environment]
    model_path = proj_path/'model_chkpts'
    # data path
    #data_path = proj_path/'TestingData'   # Test path
    #data_path = proj_path/'Ensemble_learning data'      # [CONFIRM THIS for new environment]
    data_path = proj_path.parent
    # 2 types of images (HE  FISH)
    data_name = ['HE_RBG_Corp_images']
    dataindex_fn = data_name[0]+'_processed/dataIndex(ubuntu).csv'
    dataindex_path = data_path/dataindex_fn
    #data_name = ['HE images', 'HIPT_AGH_FluorescentImage_R1']
    # 2 groups to classify
    class_names = ['Responder','NonResponder']

print(PATH_ARGS.__dict__)
def mkdirifNE(p):
    if not os.path.exists(p): os.mkdir(p)

mkdirifNE(PATH_ARGS.model_path)

def load_img(img_paths: list, is_mask=False):
        """ load array from a list of image paths """
        if is_mask: flag = 0
        else: flag = -1
        return np.concatenate([np.expand_dims(cv2.imread(str(img_fp), flag), axis=0)
                               for img_fp in img_paths.tolist()])
def normalize(ratios):
    """normalize a list of ratios to sum to 1"""
    return [r/sum(ratios) for r in ratios]

class META_ARGS:
        RANDOM_SEED = 42
        INPUT_DIM = (224,224)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DATA_ARGS:
    num_classes = 2
    batch_size = 16 if AVAIL_GPUS else 512
    n_workers = AVAIL_CPUS   # only work at 0 for strategy=None, 

def _get_normalize_attributes(data_index_df):
    x_imgs = load_img(data_index_df['x_img_path'])
    means, stds = np.mean(x_imgs, axis=((0,1,2))), np.std(x_imgs, axis=((0,1,2)))
    return means, stds

# dataloader
class HEData(Dataset):
    def __init__(self, dataindex_df: pd.DataFrame,
                 x_img_cols:str=['x_img_path'], y_cols:list=['label'],
                 transform=None, target_transform=None,
                 debug:bool=False):
        """ 
        parameters
            csv_file: contain indexer file

        """
        self.debug = debug
        # 
        self.n = len(dataindex_df)
        # fetch individual 
        self.y_ds = dataindex_df[y_cols]
        self.num_classes = self.y_ds.nunique()
        self.y_ds_enc = self.label_encode(self.y_ds, oh=False)
        # 
        self.transform = transform
        self.target_transform = target_transform
        if self.debug:
            print(f"Target shape:{self.y_ds_enc.shape}")
            print(f"[INFO]Image classes: {self.num_classes} with {self.n} instances.")

    def __len__(self):
        return self.n

    def label_encode(self, ys, oh:bool=False):
        # encode target label
        if oh:
            self.enc = OneHotEncoder()
            return self.enc.fit_transform(ys).toarray()
        else:
            self.enc = LabelBinarizer()
        ys_enc = self.enc.fit_transform(ys)
        return ys_enc.flatten()

    def __getitem__(self, idx):
        # input images
        if self.debug: print(f"Instance series: {self.y_ds.iloc[idx]},{self.y_ds.iloc[idx].name}, {idx}")
        parent_path, _, _, tile_name = self.y_ds.iloc[idx].name   # parent, type, source_tissue, tile_name
            # get data
        x_data = np.array(Image.open(Path(parent_path)/tile_name))
        y_data = self.y_ds_enc[idx]    #.reshape((-1,))
        if self.transform is not None:
            x_data = self.transform(image=x_data)['image']

        if self.target_transform:
            y_data = self.target_transform(y_data)
        # outputs g(t)
        if self.debug: print(x_data.shape, x_data.dtype, y_data.shape, y_data.dtype)
        return x_data.float(), torch.tensor(y_data, dtype=torch.long)

def get_transforms(target_size=(224,224), get_normalizing_attributes:bool=False, data_index_df:pd.DataFrame=False):    
    assert bool(get_normalizing_attributes) == bool(data_index_df), "must be provided together"
    p1 = 0.1
    p2 = 0.05
    p3 = 0.2

    if get_normalizing_attributes:
        im_means, im_stds = _get_normalize_attributes()
    else:   # use pre-computed values
        im_means, im_stds=[0, 0, 0], [1, 1, 1]
    ## Transforms
    process_transform = A.Compose([
        ToTensorV2(),
    ]) # Normalize by channel means, stds
    color_transform = A.Compose([
        # In-place transformations
        A.RandomBrightnessContrast(p=p2),
        A.RandomGamma(gamma_limit=(80, 200), p=p3),
        A.Blur(blur_limit=7, p=p2),
        A.ToGray(p=p2),
        A.CLAHE(p=p2),
        A.ChannelDropout(channel_drop_range=(1, 2), fill_value=0, p=p2),
        A.ChannelShuffle(p=p2),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2,
            always_apply=False,
            p=p2,
        ),
        A.Equalize(mode="cv", by_channels=True, mask=None, mask_params=(), p=p2),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=p2),
        A.Posterize(num_bits=4, p=p2),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=p2),
        A.GaussianBlur(blur_limit=(3, 7), p=p1)
        #A.GaussianBlur(11, sigma=(0.1, 2.0)),
    ])
    geometric_transform = A.Compose([
        A.Affine(
            scale=(0.60, 1.60),
            interpolation=cv2.INTER_LINEAR,
            cval=0,
            cval_mask=0,
            mode=cv2.BORDER_CONSTANT,
            fit_output=False,
            p=p1,
        ),
        A.Affine(
            translate_percent=(-0.2, 0.2),
            interpolation=cv2.INTER_LINEAR,
            cval=0,
            cval_mask=0,
            mode=cv2.BORDER_CONSTANT,
            fit_output=False,
            p=p1,
        ),
        A.Affine(
            rotate=(-30, 30),
            interpolation=cv2.INTER_LINEAR,
            cval=0,
            cval_mask=0,
            mode=cv2.BORDER_CONSTANT,
            fit_output=False,
            p=p1,
        ),
        A.Affine(
            shear=(-20, 20),
            interpolation=cv2.INTER_LINEAR,
            cval=0,
            cval_mask=0,
            mode=cv2.BORDER_CONSTANT,
            fit_output=False,
            p=p1,
        ),
        # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=pt),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    ###
    transformers = {'process': process_transform,  
                    'color': color_transform, 'geometric': geometric_transform}
    set_transformers = {'train': A.Compose(color_transform.transforms + geometric_transform.transforms+process_transform.transforms),
                        'val': A.Compose(process_transform.transforms),
                        'test': A.Compose(process_transform.transforms)}
    return set_transformers

# full dataset objecct
class HEDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, dataindex_path=Path('./dataIndex.csv'), label_col='label', debug=False):
        super().__init__()
        self.dataindex_path = Path(dataindex_path)
        self.batch_size = batch_size
        self.num_classes = DATA_ARGS.num_classes
        self.label_col = label_col
        self.transforms = get_transforms()
        self.debug = debug
        print(f"Debug mode:{self.debug}")
        self.index_col_len = 4

    def get_sampler(self, dataset):
        """get sampler if needed"""
        if self.label_col:
            class_cts = dataset[self.label_col].value_counts()
            for label in class_cts.index:
                class_cts.loc[label] = len(dataset)/class_cts.loc[label]
            weights = np.zeros(len(dataset))
            for label in class_cts.index:
                weights[np.where(dataset[self.label_col].to_numpy()==label)[0]] = class_cts.loc[label]
            class_balance_sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        else:
            class_balance_sampler = None
        return class_balance_sampler

    def setup(self, stage=None):
        self.datasets = dict()
        self.sampler = dict()
        # ['train', 'test', 'val']
        dataindex_df = pd.read_csv(self.dataindex_path, index_col=list(range(self.index_col_len)))
        dataindex_df = dataindex_df[dataindex_df['set'].isnull()!=True]
        for dset in dataindex_df['set'].unique():
            self.sampler[dset] = self.get_sampler(dataindex_df[dataindex_df['set']==dset])
            self.datasets[dset] = HEData(dataindex_df[dataindex_df['set']==dset],
                                         transform = self.transforms[dset], debug=self.debug)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.datasets['train'], batch_size=self.batch_size, shuffle=False if self.sampler['train'] else True, sampler=self.sampler['train'],
            num_workers=DATA_ARGS.n_workers,
        )   #persistent_workers=True) #, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.datasets['val'], batch_size=self.batch_size, shuffle=False if self.sampler['val'] else True, sampler=self.sampler['val'],
            num_workers=DATA_ARGS.n_workers,
        )  #persistent_workers=True , pin_memory=True)
        return valid_loader

from torchvision.models import EfficientNet_B7_Weights, ResNeXt101_32X8D_Weights, MobileNet_V3_Large_Weights, ResNet50_Weights

# model and train args
class MODEL_ARGS:
    n_classes = len(PATH_ARGS.class_names)

class TRAIN_ARGS:
    batch_size = DATA_ARGS.batch_size
    epochs = 100

#from torch._C import device
class HEClassificationModel(pl.LightningModule):
    def __init__(self, model_name:str, n_classes:int=2, pretrain:bool=True,
                 custom_classification_head:bool=False, input_size:tuple=(224,224), 
                 log_metrics:bool=False, debug:bool=False):
        super().__init__()
        print(f"Using pre-trained head:{model_name}")
        avail_models =  ['mobilenetv3','resnext101','efficientnetb7','resnet50']
        assert model_name in ['mobilenetv3','resnext101','efficientnetb7','resnet50'], f"Must be one of {avail_models}"
        self.debug = debug
        self.n_classes = n_classes
        self.custom_classification_head = custom_classification_head
        # Step 1: Initialize model with the weights
        if model_name == 'mobilenetv3':
            self.model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrain else None)
        elif model_name == 'resnext101':
            self.model = models.resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrain else None)
        elif model_name == 'efficientnetb7':
            self.model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrain else None)
        elif model_name =='resnet50':
            self.model = models.resnet50(pretrained=ResNet50_Weights.IMAGENET1K_V2 if pretrain else None)
        # replace/remove head
        removed = list(self.model.children())[:-1]
        self.model_base = torch.nn.Sequential(*removed)  
        in_feats = self._get_output_feat(self.model_base, input_size)
            # head
        if self.custom_classification_head:
            self.model_head = self.classification_head()
        else:
            self.model_head = nn.Sequential(nn.Flatten(),
                                            nn.Linear(in_features=in_feats, out_features=self.n_classes, bias=True),
                                            nn.ReLU(),
                                            nn.LogSoftmax(dim=1) if n_classes>2 else nn.Sigmoid(),
                                           )
        self.model = torch.nn.Sequential(self.model_base, self.model_head)
            #self.model_head.to(device=META_ARGS.device)     
        # metrics
        self.log_metrics = log_metrics
        self.sync_dist = True
        if log_metrics:
            self.metric_device = 'cpu'
            self.accuracy = torchmetrics.Accuracy().to(self.metric_device)
            self.recall = torchmetrics.Recall(average='macro', num_classes=2).to(self.metric_device)
            #self.ROC = torchmetrics.ROC(num_classes=n_classes)
            self.AUROC = torchmetrics.AUROC(num_classes=n_classes, pos_label=1).to(self.metric_device)

    def _get_output_feat(self, model, in_shape=(224,224)):
        x = torch.randn((3,)+in_shape)
        return model(x.unsqueeze(0)).flatten().size()[0]

    def _forward_feature_extract(self, x):
        return self.model_base(x)

    def forward(self, x):
        x = self.model(x)

#         #x = self.model_head(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=self.n_classes, bias=True)(x))
#         x = F.log_softmax(x, dim=1)
        #self.model.classifier = nn.Sequential(*self.model.classifier, nn.Softmax())
        if self.debug: print(f"Num classes:{self.n_classes}\nModel classifier\n:{self.model_head}")
        return x

    def add_classification_head(self):
        #n_features = self.model_head.fc.in_features
        classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.model_base.classifier[1].in_features, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512 , 256),
            nn.Linear(256 , self.n_classes),
            nn.Softmax(dim=1)
        )
        return classifier_layer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-10)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1e5)
        return [optimizer], [lr_scheduler]

    def get_loss(self, y_hat, y):
        #loss = nn.CrossEntropyLoss()   # does softmax for you (no need in classifcation)
        #loss = nn.LogSoftmax()
        #loss = F.nll_loss
        if self.debug: print(y.size(), y.dtype, y_hat.size(), y_hat.dtype)
        return F.cross_entropy(y_hat,  y)

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        # training metrics
        
        # optimize (done under the hoood)
        if self.log_metrics:
            acc = self.accuracy(torch.argmax(y_hat, dim=1).to(self.metric_device), y.to(self.metric_device))
            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=self.sync_dist)
            self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=self.sync_dist)
        return loss
        #return self.get_loss(y, y_hat)

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        # compute metrics
        val_loss =self.get_loss(y_hat, y)
        if self.log_metrics:
            acc = self.accuracy(torch.argmax(y_hat, dim=1).to(self.metric_device).detach(), y.to(self.metric_device).detach())
            #auroc = self.AUROC(y_hat.to(self.metric_device), y.to(self.metric_device))
            #fpr, tpr, thresholds = self.ROC(y_hat, y)

            self.log("val_loss", val_loss)
            self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=self.sync_dist)
            #self.AUROC.update(y_hat.cpu().detach(), y.cpu().detach())
            #self.log("validation_auc", self.AUROC, on_step=False, on_epoch=True, sync_dist=self.sync_dist)   # prog_bar=True,


class HEEnsembleModel(pl.LightningModule):
    def __init__(self, 
                 ensembles_settings:dict={'efficientnetb7':3, 'resnext101':2}, 
                 pretrain:bool=True,
                 n_classes:int=2,
                 input_shape=(224,224),
                 metrics={},
                 debug=False):
        super(HEEnsembleModel, self).__init__()
        self.debug = debug
        self.sync_dist = True
        models = []
        self.n_models = 0
        for name, number in ensembles_settings.items():
            [models.append(
                HEClassificationModel(model_name=name, 
                                      n_classes=2, 
                                      pretrain=pretrain,
                                      log_metrics=False,
                                      custom_classification_head=False
                                     )
                         ) for i in range(number)
            ]
            self.n_models += number
        self.ensemble_model = torch.nn.ModuleList(models)
        self.classifier = torch.nn.Linear(self.n_models*n_classes, n_classes)
        #self.save_hyperparameters() # Uncomment to show error
        self.CEloss = nn.CrossEntropyLoss()
        # metrics
        self.metrics = metrics
        self.metric_device = 'cuda:1'
        #self.AUROC = torchmetrics.AUROC(num_classes=n_classes, pos_label=1)

    def forward(self, x):
        output=[]
        for m in self.ensemble_model:
            output.append(m(x))
        combined = torch.concat(output,dim=1)
        x = self.classifier(combined)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-10)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def get_loss(self, y_hat, y):
        #loss = nn.CrossEntropyLoss()   # does softmax for you (no need in classifcation)
        #loss = nn.LogSoftmax()
        #loss = F.nll_loss
        if self.debug: print(y.size(), y.dtype, y_hat.size(), y_hat.dtype)
        return self.CEloss(y_hat,  y)

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        # training metrics
        self.accuracy = torchmetrics.Accuracy().to(self.metric_device)
        self.recall = torchmetrics.Recall(average='macro', num_classes=2).to(self.metric_device)
        acc = self.accuracy(torch.argmax(y_hat, dim=1).to(self.metric_device), y.to(self.metric_device))
        rec = self.recall(torch.argmax(y_hat, dim=1).to(self.metric_device), y.to(self.metric_device))
        # optimize (done under the hoood)

        self.log('train_loss', loss, on_step=True, on_epoch=True,  sync_dist=self.sync_dist)
        self.log('train_acc', acc, on_epoch=True, sync_dist=self.sync_dist)
        self.log('train_rec', rec, on_epoch=True,  sync_dist=self.sync_dist)
        return loss
        #return self.get_loss(y, y_hat)

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        # compute metrics
        val_loss =self.get_loss(y_hat, y)
        self.accuracy = torchmetrics.Accuracy().to(self.metric_device)
        self.recall = torchmetrics.Recall(average='macro', num_classes=2).to(self.metric_device)
        acc = self.accuracy(torch.argmax(y_hat, dim=1).to(self.metric_device), y.to(self.metric_device))
        rec = self.recall(torch.argmax(y_hat, dim=1).to(self.metric_device), y.to(self.metric_device))
        if hasattr(self, 'AUROC'):
            auroc = self.AUROC(y_hat.to(self.metric_device), y.to(self.metric_device))
        #fpr, tpr, thresholds = self.ROC(y_hat, y)
        #
        self.log("val_loss", val_loss, on_step=True, sync_dist=self.sync_dist)
        self.log('val_acc', acc,  on_epoch=True, sync_dist=self.sync_dist)
        self.log('val_rec', rec,  on_epoch=True, sync_dist=self.sync_dist)
        if hasattr(self, 'AUROC'):
            self.AUROC.update(y_hat.to(self.metric_device), y.to(self.metric_device))
            self.log("validation_auc", auroc, on_step=False, on_epoch=True, sync_dist=self.sync_dist)  # prog_bar=True
        #self.log("val_auc", valid_auc, on_step=False, on_epoch=True, prog_bar=True)


# In[32]:


# callbacks
class PRMetrics(pl.Callback):
    """ Custom callback to compute per-class PR & ROC curves
    at the end of each training epoch
    """
    def __init__(self,  val_samples, num_samples=32, class_names={'Non-responder':0, 'Responder':1}):    #generator=None, num_log_batches=1):
        # self.generator = generator
        # self.num_batches = num_log_batches
        # # store full names of classes
        # self.class_names = { v: k for k, v in generator.class_indices.items() }
        # self.flat_class_names = [k for k, v in generator.class_indices.items()]

        super().__init__()
        self.num_samples = num_samples
        self.class_names = class_names
        self.val_imgs, self.val_labels = val_samples

    def on_epoch_end(self, trainer, pl_module, logs={}):
        # # collect validation data and ground truth labels from generator
        # val_data, val_labels = zip(*(self.generator[i] for i in range(self.num_batches)))
        # val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)

        # # use the trained model to generate predictions for the given number
        # # of validation data batches (num_batches)
        # val_predictions = self.model.predict(val_data)
        # ground_truth_class_ids = val_labels.argmax(axis=1)
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        preds = torch.argmax(pl_module(val_imgs), -1)
        # Log precision-recall curve the key "pr_curve" is the id of the plot--do not change this if you want subsequent runs to show up on the same plot
        wandb.log({"roc_curve" : wandb.plot.roc_curve(val_labels, preds, labels=self.class_names)})

class ImagePredictionLogger(pl.Callback):
    """ callback"""
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
        })
        
def main():
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import WandbLogger
    #
    data_index_df = pd.read_csv(PATH_ARGS.dataindex_path, index_col=list(range(4)))

    # DEFAULT (ie: no accumulated grads)
    cbs = [
        pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=PATH_ARGS.model_path,
                                     filename='models-{epoch:02d}-{val_loss:.2f}', save_top_k=2, mode='min'),
        pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-7, patience=8, mode="min"),
        pl.callbacks.GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1}),
        #PRMetrics(),
    ]
    
    trainer = pl.Trainer(
        accelerator="auto",   #"gpu",
        devices='auto',   #2,
        logger=WandbLogger(project='AD-ensemble(draft)',  entity="3m-m", job_type='train'),
        max_epochs=TRAIN_ARGS.epochs, callbacks=cbs,
        strategy='dp', # "horovod",  # dp
        #plugins=DDPPlugin(find_unused_parameters=True),
    )
    
    model = HEEnsembleModel(
        ensembles_settings={'efficientnetb7':1, 'mobilenetv3':1, 'resnext101':1},
        pretrain=False,
        input_shape=(224,224),
        n_classes=MODEL_ARGS.n_classes,
        debug=False
    )

    # train
    datamodule = HEDataModule(batch_size=TRAIN_ARGS.batch_size, dataindex_path=PATH_ARGS.dataindex_path, debug=False)
    datamodule.setup()
    trainer.fit(model=model, datamodule=datamodule) 
    # save with parameters


# ### Prediction/submission
#test_loader = DataLoader(HEdatasets['test'], shuffle=True, batch_size=TRAIN_args.test_batch_size)

if __name__ == "__main__":
    import datetime
    print(f"Starting run at:{datetime.datetime.now()}")
    main()