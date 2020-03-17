import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from pathlib import Path
import albumentations as A
# from .dataset import MNISTDataset


class Model(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_transforms(self):
        return A.Normalize()

    @staticmethod
    def _prepare_batch(img):
        img = np.moveaxis(img, -1, 0)
        vec = torch.from_numpy(img)
        batch = torch.unsqueeze(vec, 0)
        return batch

    def summarize(self, ds):
        train_path = Path(ds)
        imgs = list(train_path.glob('**/*.png'))
        dataset = MNISTDataset(paths=imgs,
                               mode="train",
                               transforms=self.get_transforms())
        acc = {True: 0, False: 0}
        for item in torch.utils.data.DataLoader(dataset):
            result = self(item["features"])
            result = result.detach().cpu().numpy()
            acc[(item["targets"] == result.argmax()).item()] += 1

        print(len(dataset))
        print(acc[True]/len(dataset))


CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


class MNISTDataset(Dataset):
    """ dataset to read mnist png data, apply transforms if there any,
        and return with corresponded class taken from image path

        args:
        paths: paths of images
        mode: mode of dataset usage to understand if targets
            need to be extracted or not
        transforms (a.compose): data transformation pipeline
            from albumentations package (e.g. flip, scale, etc.)
    """

    def __init__(self, paths: list, mode: str, transforms=None):
        self.paths = paths
        mode_err = "Mode should be one of `train`, `valid` or `infer`"
        assert mode in ["train", "valid", "infer"], mode_err
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # We need to cast Path instance to str
        # as cv2.imread is waiting for a string file path
        item = {"paths": str(self.paths[idx])}
        img = cv2.imread(item["paths"])
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        img = np.moveaxis(img, -1, 0)
        item["features"] = torch.from_numpy(img)

        if self.mode != "infer":
            # We need to provide a numerical index of a class, not string,
            # so we cast it to int
            item["targets"] = int(item["paths"].split("/")[-2])

        return item
