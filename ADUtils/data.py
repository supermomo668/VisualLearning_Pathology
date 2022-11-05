from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch 

import pytorch_lightning as pl
import albumentations as A, cv2
from albumentations.pytorch.transforms import ToTensorV2 

import pandas as pd , numpy as np, PIL.Image as Image
from pathlib import Path
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split

class DATA_ARGS:
    n_workers = 4   # only work at 0 for strategy=None, 

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
        self.y_ds_enc = self.label_encode(self.y_ds, oh=False)
        # 
        self.transform = transform
        self.target_transform = target_transform
        if self.debug:
            print(f"Target shape:{self.y_ds_enc.shape}")
            print(f"[INFO]{self.n} instances.")

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
    
# full dataset objecct
class HEDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, dataindex_path=Path('./dataIndex.csv'), label_col='label', 
                 patch_size=224, debug=False):
        super().__init__()
        self.dataindex_path = Path(dataindex_path)
        self.batch_size = batch_size
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
    def custom_collate(self, batch):
        return torch.cat(batch)
    
    def train_dataloader(self):
        train_loader = DataLoader(
            self.datasets['train'], batch_size=self.batch_size, shuffle=False if self.sampler['train'] else True, sampler=self.sampler['train'],
            num_workers=64, pin_memory=True, collate_fn=self.custom_collate
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.datasets['val'], batch_size=self.batch_size, shuffle=False if self.sampler['val'] else True, sampler=self.sampler['val'],
            num_workers=64, pin_memory=True, collate_fn=self.custom_collate
        )
        return valid_loader