import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm
import skimage.io
import skimage.segmentation


class CellDataset(Dataset):
    def __init__(self, path, transform=None, train=True):
        self.path = path
        self.transform = transform
        self.imgs = os.listdir(path)
        self.train = train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.imgs[idx], 'images', self.imgs[idx] + '.png')
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.train:
            mask_paths = [os.path.join(self.path, self.imgs[idx], 'masks', filename) for filename in
                          os.listdir(os.path.join(self.path, self.imgs[idx], 'masks'))]
            mask = torch.zeros_like(img)
            for i, mask_path in enumerate(mask_paths):
                mask_ = read_image(mask_path)
                mask[mask_ != 0] = 1 * (i + 1)

            return img, mask

        return img, None


def hover_array(path):
    """
    Create numpy array suitable for hover input based on path to directory
    containing image and masks
    """
    img_path = os.path.join(os.path.join(path, "images",os.listdir(os.path.join(path, "images"))[0]))
    # https://github.com/vqdang/hover_net/tree/master?tab=readme-ov-file#data-format
    hov_input = plt.imread(img_path)
    # in train set images had 4th channel, but in test set not
    # in train set alpha channel was always 1, ommiting 4th channel
    hov_input = hov_input[:,:,:3]
    # adding channel for instance labels
    hov_input = np.concatenate((hov_input, np.expand_dims(np.zeros(hov_input.shape[:2]), 2)), axis=2)
    masks_path = os.path.join(path, "masks")

    # for each mask, add mask's idx + 1 to the 4th channel 
    # assumes that pxels with value 1 in masks do not overlap
    for idx, mask_file in enumerate(os.listdir(masks_path)):
        mask_array = plt.imread(os.path.join(masks_path, mask_file))
        mask_array = mask_array * (idx + 1)
        hov_input[:,:,3] += mask_array
    
    return hov_input


def create_hover_data(path, input_path=None):
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
    for img_name in tqdm(os.listdir(path)):
        example_path = os.path.join(path, img_name)
        hov_array = hover_array(example_path)
        np.save(os.path.join(input_path, f"{img_name}.npy"), hov_array)

def encoded_pixels_to_mask(pixels_str, shape):
    """
    Parses string with encoded pixels, returns mask (one nuclei) in a form of numpy array
    """
    pixels_str = pixels_str.strip().split(" ")
    ranges_dict = {int(pixels_str[i]): int(pixels_str[i+1]) for i in range(0, len(pixels_str)-1, 2)}
    arr = np.zeros(shape).flatten()
    # print(arr)

    for pixel_start, pixel_range in ranges_dict.items():
        for i in range(pixel_range):
            arr[pixel_start + i - 1] = 1

    # plt.imshow(np.reshape(arr, shape))
    # plt.plot()
    return(np.transpose(np.reshape(arr, shape)))

def labels_csv_to_masks(labels_csv, input_data_dir, out_dir):
    """
    Parses csv with encoded pixels and based on the pictures in input_data_dir creates masks.
    Masks are saved in the out_dir, one per nuclei.

    Parameters:
        labels_csv:
            path to csv with encoded pixels

        input_data_dir:
            path to directory with directories with names (and images) corresponding to ids in
            labels_csv

        out_dir:
            directory the encoded masks will be written to
    """
    df_labels = pd.read_csv(labels_csv)

    shapes_dict = {}
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    for img_dir in os.listdir(input_data_dir):
        img_path = os.path.join(input_data_dir, img_dir, "images")
        assert len(os.listdir(img_path)) == 1
        shapes_dict[img_dir] = plt.imread(os.path.join(input_data_dir, img_dir, "images", img_dir+".png")).shape
        mask_dir = os.path.join(out_dir, img_dir)
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)

    masks_count = {img_id: 0 for img_id in shapes_dict.keys()}
    for idx, img_row in df_labels.iterrows():
        img_id = img_row["ImageId"]
        try:
            img_shape = shapes_dict[img_id]
        except KeyError:
            print(f"Input picture with id {img_id} not found in the provided directory")
            continue
        mask_arr = encoded_pixels_to_mask(img_row["EncodedPixels"], img_shape[:2])
        matplotlib.image.imsave(os.path.join(out_dir, img_id, f"mask_{masks_count[img_id]}.png"), mask_arr)
        masks_count[img_id] += 1
        
def crop_to_size(img, save_path):
    shape = img.shape
    crop = np.zeros((256, 256, 4))
    for i in tqdm([k * 245 for k in range(shape[0] // 245)], desc=f"Saving to {save_path}", leave=False):
        for j in [k * 245 for k in range(shape[1] // 245)]:
            crop_ = img[i:i + 256, j:j + 256, ]
            # plt.imshow(crop_)
            # plt.show()
            crop[:crop_.shape[0], :crop_.shape[1], ] = crop_
            # plt.imshow(crop)
            # plt.show()
            print(crop_.shape, crop.shape)
            np.save(save_path[:-4] + '(' + str(i) + ',' + str(j) + ').npy', crop)


def crop_path(path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for img_name in os.listdir(path):
        img = np.load(os.path.join(path, img_name))
        crop_to_size(img, os.path.join(save_path, img_name))

def mAP_iou(labels, y_pred):
    fig = plt.figure()
    plt.imshow(labels)
    plt.title("Ground truth masks")
    
    
    # Simulate an imperfect submission
    # offset = 2 # offset pixels
    # y_pred = labels[offset:, offset:]
    # y_pred = np.pad(y_pred, ((0, offset), (0, offset)), mode="constant")
    # y_pred[y_pred == 20] = 0 # Remove one object
    # y_pred, _, _ = skimage.segmentation.relabel_sequential(y_pred) # Relabel objects
    
    # Show simulated predictions
    fig = plt.figure()
    plt.imshow(y_pred)
    plt.title("Simulated imperfect submission")
    
    # Compute number of objects
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    print("Number of true objects:", true_objects - 1)
    print("Number of predicted objects:", pred_objects - 1)
    
    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
    # print(intersection)
    
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    
    # Compute union
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9
    
    # Compute the intersection over union
    iou = intersection / union
    
    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn
    
    # Loop over IoU thresholds
    prec = []
    print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))