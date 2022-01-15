from PIL import Image
from torch.utils.data import Dataset
import os
from glob import glob
import torch
import torchvision.transforms as transforms

''' Center cropped the Images such that maximum size is extracted and then resized it accordingly. '''

class inaturalist(Dataset):
    def __init__(self, root_dir, mode = 'train', transform = True):
        self.data_dir = root_dir
        self.mode = mode
        self.transforms = transform      
        self._init_dataset()

    def _init_dataset(self):
        self.files = []
        self.labels = []
        dirs = sorted(os.listdir(os.path.join(self.data_dir, 'train')))
        if self.mode == 'train': 
            for dir in range(len(dirs)):
                files = sorted(glob(os.path.join(self.data_dir, 'train', dirs[dir], '*.jpg')))
                self.labels += [dir]*len(files)            
                self.files += files
        elif self.mode == 'val':
            for dir in range(len(dirs)):
                files = sorted(glob(os.path.join(self.data_dir, 'val', dirs[dir], '*.jpg')))
                self.labels += [dir]*len(files)            
                self.files += files
        else:
            print("No Such Dataset Mode")
            return None
    
    def _init_transform(self, crop_size):  
        # All images are of different size, so resize is necessary
        self.transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                             transforms.Resize([128, 128]),
                                             transforms.ToTensor(), 
                                             transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
    
    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB')
        label = self.labels[index]
                    
        if self.transforms:
            height, width = img.size
            crop_size = min(height, width)
            self._init_transform(crop_size)
            img = self.transform(img)
            
        label = label - 1

        label = torch.tensor(label, dtype = torch.long)

        return img, label

    def __len__(self):
        return len(self.files)
