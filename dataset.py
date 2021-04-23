import torch
import PIL
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class ToyDataset(torch.utils.data.Dataset): 
    def __init__(self, data_mat):
        self.data = data_mat
        self.z_dim = len(data_mat[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]

class CelebA(torch.utils.data.Dataset):
    def __init__(self, data_home):
        self.data_home = '%s/img_align_celeba' % data_home
        self.x_list = np.load('%s/x_list.npy' % data_home)
        # self.list_attr = pd.read_csv('%s/list_attr_celeba.csv' % data_home)
        # self.list_bbox = pd.read_csv('%s/list_bbox_celeba.csv' % data_home)
        # self.list_eval = pd.read_csv('%s/list_eval_partition.csv' % data_home)

        # closecrop
        self.transform = transforms.Compose([
            transforms.CenterCrop((140, 140)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),])
        
    def __len__(self):
        return len(self.x_list)
    
    def __getitem__(self, idx):
        with Image.open('%s/%s' % (self.data_home, self.x_list[idx])) as im:
            return self.transform(im)