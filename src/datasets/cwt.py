import os
from os.path import dirname, join, realpath
from .utils import *
from .base_dataset import BaseDataset
from PIL import Image

__all__ = [
    'data_dir',
    'CWT',
]

data_dir = realpath(join(dirname(__file__), '..', '..', 'data'))


class CWT(BaseDataset):
    CLASSES = ["flat", "bumpy", "water", "rock", "mixed", "excavator", "obstacle"]
    PALETTE = [[0, 255, 0], [255, 255, 0], [255, 0, 0], [128, 0, 0], [100, 65, 0], [0, 255, 255], [0, 0, 255]]

    def __init__(self,
                 path=None,
                 split='train',
                 num_samples=None,
                 classes=None,
                 multi_scale=True,   # TODO: fix padding, background must be black for masks (0)
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(1200, 1920),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=np.asarray([0.0, 0.0, 0.0]),
                 std=np.asarray([1.0, 1.0, 1.0])):
        super(CWT, self).__init__(ignore_label, base_size, crop_size, downsample_rate, scale_factor, mean, std, )

        if path is None:
            path = join(data_dir, 'CWT')
        assert os.path.exists(path)
        assert split in ['train', 'test', 'val']
        # validation dataset is called 'test'
        if split == 'val':
            split = 'test'
        self.path = path
        self.split = split
        if not classes:
            classes = self.CLASSES
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.palette_values = [self.PALETTE[c] for c in self.class_values]
        self.color_map = {}
        for k, v in zip(self.class_values, self.palette_values):
            self.color_map[k] = v

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1. / downsample_rate

        self.multi_scale = multi_scale
        self.flip = flip

        self.fps = np.genfromtxt(os.path.join(path, '%s.txt' % split), dtype=str).tolist()

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for path in self.fps:
            name = path.split('/')[1]
            files.append({
                "img": os.path.join(self.path, 'img', '%s.jpg' % path),
                "label": os.path.join(self.path, 'annotation/grey_mask', '%s.png' % path),
                "name": name,
            })
        return files

    def __getitem__(self, index):
        item = self.files[index]
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)

        mask = np.array(cv2.imread(item["label"], 0))

        # add augmentations
        image, mask = self.apply_augmentations(image, mask, self.multi_scale, self.flip)

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=0).astype('float')

        return image.copy(), mask.copy()


def demo():
    from datasets.utils import visualize, convert_color

    # split = np.random.choice(['test', 'train', 'val'])
    split = 'train'
    ds = CWT(split=split)

    for _ in range(5):
        image, gt_mask = ds[int(np.random.choice(range(len(ds))))]
        image = image.transpose([1, 2, 0])
        image_vis = np.uint8(255 * (image * ds.std + ds.mean))

        gt_arg = np.argmax(gt_mask, axis=0).astype(np.uint8)
        gt_color = convert_color(gt_arg, ds.color_map)

        visualize(
            image=image_vis,
            label=gt_color,
        )


def main():
    demo()


if __name__ == '__main__':
    main()
