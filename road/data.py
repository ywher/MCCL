"""Road-domain loaders; unlabeled targets never open their annotation paths."""
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .transforms import blur, crop, hflip, normalize, obtain_cutmix_box, resize


def read_ids(path):
    ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f'Empty split: {path}')
    return ids


def parse_id(line):
    parts = line.split(',', 1) if ',' in line else line.split(None, 1)
    return parts[0].strip(), parts[1].strip() if len(parts) == 2 else None


class RoadDataset(Dataset):
    def __init__(self, root, id_path, mode, nclass, size=1024, pre_resize=None, nsample=None):
        if mode not in ('train_l', 'train_u', 'val'):
            raise ValueError(mode)
        self.root, self.mode, self.nclass = Path(root), mode, nclass
        self.size, self.pre_resize = size, pre_resize
        self.ids = read_ids(id_path)
        if nsample is not None and nsample > len(self.ids):
            self.ids = (self.ids * math.ceil(nsample / len(self.ids)))[:nsample]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_path, mask_path = parse_id(self.ids[index])
        with Image.open(self.root / image_path) as f:
            image = f.convert('RGB')
        if self.mode == 'train_u':
            mask = Image.new('L', image.size, 0)
        else:
            if mask_path is None:
                raise ValueError(f'Missing label in {self.ids[index]}')
            with Image.open(self.root / mask_path) as f:
                values = np.array(f)
            if values.ndim != 2 or np.any((values != 255) & (values >= self.nclass)):
                raise ValueError(f'Expected preconverted {self.nclass}-class trainIds: {mask_path}')
            mask = Image.fromarray(values.astype(np.uint8))
        if self.pre_resize:
            h, w = self.pre_resize
            image = image.resize((w, h), Image.Resampling.BILINEAR)
            mask = mask.resize((w, h), Image.Resampling.NEAREST)
        if self.mode == 'val':
            image, mask = normalize(image, mask)
            return image, mask, self.ids[index]
        image, mask = resize(image, mask, (1.0, 2.0) if self.pre_resize else (0.5, 2.0))
        image, mask = crop(image, mask, self.size, 254 if self.mode == 'train_u' else 255)
        image, mask = hflip(image, mask, p=0.5)
        if self.mode == 'train_l':
            return normalize(image, mask)
        strong = []
        boxes = []
        for _ in range(2):
            view = deepcopy(image)
            if random.random() < 0.8:
                view = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(view)
            view = transforms.RandomGrayscale(p=0.2)(view)
            view = blur(view, p=0.5)
            boxes.append(obtain_cutmix_box(self.size, p=0.5))
            strong.append(normalize(view))
        ignore = torch.zeros(self.size, self.size, dtype=torch.long)
        ignore[torch.from_numpy(np.array(mask)) == 254] = 255
        return normalize(image), *strong, ignore, *boxes


class BatchStream:
    """Restart finite loaders without caching augmented batches in memory."""
    def __init__(self, loader):
        if len(loader) == 0:
            raise ValueError('Training split is smaller than its batch size')
        self.loader, self.epoch, self.iterator = loader, 0, iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.iterator = iter(self.loader)
            return next(self.iterator)
