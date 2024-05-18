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


def create_hover_data(path):
    """
    For HoVerNet specific type of data is needed:
    5 channels in npy file: first 3 RGB, instance label, instance type for each cell
    """
    pass
