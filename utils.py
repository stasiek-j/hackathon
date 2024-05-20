import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os


class CellDataset(Dataset):
    def __init__(self, path, transform=None, train=True):
        self.path = path
        self.transform = transform
        self.imgs = os.listdir(path)
        self.train = train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.imgs[idx], 'images', self.imgs[idx]+'.png')
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.train:
            mask_paths = [os.path.join(self.path, self.imgs[idx], 'masks', filename) for filename in
                          os.listdir(os.path.join(self.path, self.imgs[idx], 'masks'))]
            mask = torch.zeros_like(img)
            for i, mask_path in enumerate(mask_paths):
                mask_ = read_image(mask_path)
                mask[mask_ != 0] = 1 * (i+1)

            return img, mask

        return img, None

def hover_array(path):
    """
    Create numpy array suitable for hover input based on path to directory
    containing image and masks
    """
    img_path = os.path.join(os.path.join(path, "images",os.listdir(os.path.join(path, "images"))[0]))
    # pngs in a dataset contain alpha channel (always all ones) - we will use it for instance labels
    # https://github.com/vqdang/hover_net/tree/master?tab=readme-ov-file#data-format
    hov_input = plt.imread(img_path)
    # setting 0 in 4th channel as background
    masks_path = os.path.join(path, "masks")
    hov_input[:,:,3] = 0

    # for each mask, add mask's idx + 1 to the 4th channel 
    # assumes that pxels with value 1 in masks do not overlap
    for idx, mask_file in enumerate(os.listdir(masks_path)):
        mask_array = plt.imread(os.path.join(masks_path, mask_file))
        mask_array = mask_array * (idx + 1)
        hov_input[:,:,3] += mask_array
    
    return hov_input

    
def create_hover_data(path, input_path = None):
    """
    For HoVerNet specific type of data is needed:
    4 channels in npy file: first 3 RGB and instance label for each cell

    Parameters:
        path: str
            path to the train dataset
        input_path: str
            path the resulting .npy suitable for hover will be saved in
    """
    if not input_path:
        input_path = os.path.join("hover_input", path)
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    for img_name in os.listdir(path):
        example_path = os.path.join(path, img_name)
        hov_array = hover_array(example_path)
        np.save(os.path.join(input_path, f"{img_name}.npy"), hov_array)