import os

import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import crop

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_all_files(path, prefix="", suffix="", contains=""):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")
    files = []
    for pre, dirs, basenames in os.walk(path):
        for name in basenames:
            if name.startswith(prefix) and name.endswith(suffix) and contains in name:
                files.append(os.path.join(pre, name))
    return files


class MyIterableDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def process_data(self, generator):
        for frame, label, mask in generator:
            frame = torch.from_numpy(frame.numpy()).permute(0, 3, 1, 2)  # BHWC->BCHW
            label = torch.from_numpy(label.numpy()).long()
            mask = torch.from_numpy(mask.numpy())
            yield frame, label, mask

    def get_stream(self, generator):
        return self.process_data(generator)

    def __iter__(self):
        return self.get_stream(self.generator)

def tfrecords_to_dataloader(tfrec_path, feature=None, parse_example_fn=None, batch_size=12):
    tfds = tf.data.TFRecordDataset(tfrec_path)

    if feature is None:
        feature = {
            "frame": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "mask": tf.io.FixedLenFeature([], tf.string),
        }

    if parse_example_fn is None:
        def parse_example_fn(example_proto):
            parsed_feature = tf.io.parse_single_example(example_proto, feature)
            frame = tf.io.parse_tensor(parsed_feature["frame"], tf.uint8)
            frame = tf.cast(tf.reshape(frame, [1080, 1920, 3]), tf.float32)
            label = tf.cast(parsed_feature["label"], tf.int64)

            mask = tf.io.parse_tensor(parsed_feature["mask"], tf.uint8)
            mask = tf.cast(tf.reshape(mask, [1080, 1920]), tf.float32)

            return frame, label, mask

    return tfds.map(parse_example_fn).batch(batch_size).prefetch(AUTOTUNE)


class GenericImageDataset(Dataset):
    def __init__(self, img_paths, mask_available=True, return_labels=True):
        self.img_paths = img_paths
        self.mask_available = mask_available
        self.return_labels = return_labels
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def _disect_path(self, path):
        folder, basename = os.path.split(os.path.abspath(path))
        filename, extension = os.path.splitext(basename)
        return folder, filename, extension

    def _get_input(self, path):
        try:
            return read_image(path).to(torch.float32)[0:3]
        except:
            print(f"can't open {path}")
            return torch.zeros(3, 1080, 1920)

    def _get_label(self, filename):
        if "orig" in filename:
            return 0
        else:
            return 1

    def _get_mask(self, folder, filename):
        mask = read_image(f"{folder}/{filename}.mask").squeeze()
        mask = (mask / 255.0).to(torch.uint8)
        return mask

    def __getitem__(self, index):
        path = self.img_paths[index]
        folder, filename, extension = self._disect_path(path)
        img = self._get_input(path)
        label = self._get_label(path)

        if not (self.return_labels or self.mask_available):
            return img

        batch = [img]
        if self.return_labels:
            batch.append(label)
        if self.mask_available:
            if label:
                mask = self._get_mask(folder, filename)
            else:
                mask = torch.zeros(1080, 1920).to(torch.uint8)
            batch.append(mask)

        return batch


class CarvalhoImageDataset(GenericImageDataset):
    def _get_input(self, path):
        img = read_image(path).to(torch.float32)[0:3]
        if img.shape[1] > img.shape[2]:
            img = img.permute(0, 2, 1)
        if img.shape[1] != 1080:
            img = crop(img, 0, 0, 1080, 1920)
        return img

    def _get_label(self, filename):
        if "normal" in filename:
            return 0
        else:
            return 1

    def _get_mask(self, folder, filename):
        folder = folder.replace("carvalho_tampered", "carvalho_masks")
        path = get_all_files(f"{folder}", prefix=f"{filename}_", suffix=".png")[0]
        mask = read_image(path)
        mask = 1 - (mask.sum(dim=0) / (255.0 * mask.shape[0]))
        if mask.shape[0] > mask.shape[1]:
            mask = mask.permute(1, 0)
        if mask.shape[0] != 1080:
            mask = crop(mask, 0, 0, 1080, 1920)
        return mask


class KorusImageDataset(GenericImageDataset):
    def _get_input(self, path):
        img = Image.open(path, mode="r")
        img = self.to_tensor(img) * 255
        if img.shape[1] > img.shape[2]:
            img = img.permute(0, 2, 1)
        if img.shape[1] != 1080:
            img = crop(img, 0, 0, 1080, 1920)
        return img

    def _get_label(self, filename):
        if "normal" in filename:
            return 0
        else:
            return 1

    def _get_mask(self, folder, filename):
        folder = folder.replace("korus_tampered", "korus_masks")
        path = get_all_files(f"{folder}", prefix=f"{filename}.", suffix=".PNG")[0]
        mask = read_image(path)
        mask = mask.sum(dim=0) / (255.0 * mask.shape[0])
        if mask.shape[0] > mask.shape[1]:
            mask = mask.permute(1, 0)
        if mask.shape[0] != 1080:
            mask = crop(mask, 0, 0, 1080, 1920)
        return mask


class ColumbiaImageDataset(GenericImageDataset):
    def _get_input(self, path):
        img = Image.open(path, mode="r")
        img = self.to_tensor(img) * 255
        if img.shape[1] > img.shape[2]:
            img = img.permute(0, 2, 1)
        return img

    def _get_label(self, filename):
        if "auth" in filename:
            return 0
        else:
            return 1

    def _get_mask(self, folder, filename):
        folder = folder.replace("4cam_auth", "4cam_masks")
        path = get_all_files(f"{folder}", prefix=f"{filename}.", suffix=".jpg")[0]
        mask = read_image(path)
        mask = mask.sum(dim=0) / (255.0 * mask.shape[0])
        if mask.shape[0] > mask.shape[1]:
            mask = mask.permute(1, 0)
        return mask


# Huh's In The Wild dataset -- only edited images
class ITWImageDataset(GenericImageDataset):
    def _get_input(self, path):
        img = read_image(path).to(torch.float32)[0:3]
        if img.shape[1] > img.shape[2]:
            img = img.permute(0, 2, 1)
        if img.shape[1] != 1080:
            print(img.shape)
            img = crop(img, 0, 0, 1080, 1920)

        return img

    def _get_label(self, filename):
        return 1

    def _get_mask(self, folder, filename):
        folder = folder.replace("huh_in_the_wild_tampered", "huh_in_the_wild_masks")
        path = get_all_files(f"{folder}", prefix=f"{filename}.", suffix=".png")[0]
        mask = read_image(path)
        mask = mask.sum(dim=0) / (255.0 * mask.shape[0])
        if mask.shape[0] > mask.shape[1]:
            mask = mask.permute(1, 0)
        if mask.shape[0] != 1080:
            mask = crop(mask, 0, 0, 1080, 1920)
        return mask
