import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[2]

def path(value):
    p=Path(value)
    return p if p.is_absolute() else ROOT/p

def read_rows(filename):
    with path(str(filename)).open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not rows: raise ValueError(f'Empty manifest: {filename}')
    return rows

def load_pair(row):
    with Image.open(path(row['image_path'])) as im:
        if im.mode.startswith('I') or im.mode=='F':
            raise ValueError('16/32-bit or floating input requires an explicit intensity policy: '+row['image_path'])
        image=np.array(im.convert('L'),dtype=np.uint8)
    with Image.open(path(row['mask_path'])) as im:
        mask=np.array(im)
    if mask.ndim!=2 or mask.shape!=image.shape:
        raise ValueError('Raw mask must be 2D and match image: '+row['mask_path'])
    if not np.isin(mask,[0,1,2]).all():
        raise ValueError('Mask IDs must be 0,1,2: '+row['mask_path'])
    if row.get('crop_x','')!='':
        x,y=int(row['crop_x']),int(row['crop_y'])
        image=image[y:y+512,x:x+512]; mask=mask[y:y+512,x:x+512]
    if image.shape!=(512,512):
        raise ValueError('Expected 512x512 input after crop: '+row['image_path'])
    return image,mask.astype(np.int64)

class PatchDataset(Dataset):
    def __init__(self,rows,augment=False): self.rows=rows; self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        image,mask=load_pair(self.rows[index]); image=image.astype(np.float32)/255.
        if self.augment:
            k=np.random.randint(4); image=np.rot90(image,k); mask=np.rot90(mask,k)
            if np.random.rand()<.5: image=np.fliplr(image); mask=np.fliplr(mask)
            if np.random.rand()<.5: image=np.flipud(image); mask=np.flipud(mask)
            if np.random.rand()<.8:
                image=np.clip((image-.5)*np.random.uniform(.85,1.15)+.5+np.random.uniform(-.07,.07),0,1)
            if np.random.rand()<.25:
                image=np.clip(image+np.random.normal(0,.015,image.shape),0,1)
        return torch.from_numpy(np.ascontiguousarray(image[None],dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(mask)),index
