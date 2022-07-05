import os
import cv2
import torch
import torch.cuda
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from traversability_estimation.utils import visualize
from traversability_estimation.rellis_3d import DatasetSemSeg as Dataset

VERBOSE = True

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

dims = {"width": 384, "height": 640}


def get_training_augmentation():
    train_transform = A.Compose(
        [
            A.Resize(dims["width"], dims["height"], interpolation=1, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            A.GaussNoise(p=0.2),
            A.Perspective(p=0.5),
            A.OneOf([A.CLAHE(p=1), A.RandomBrightness(p=1), A.RandomGamma(p=1)], p=0.9),
            A.OneOf([A.Sharpen(p=1), A.Blur(blur_limit=3, p=1), A.MotionBlur(blur_limit=3, p=1)], p=0.9),
            A.OneOf([A.RandomBrightnessContrast(p=1), A.HueSaturationValue(p=1)], p=0.9)
        ])
    return train_transform


def get_validation_augmentation():
    test_transform = A.Compose([A.Resize(dims["width"], dims["height"], interpolation=1, always_apply=True)])
    return test_transform


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fnc):
    _transform = A.Compose([A.Lambda(image=preprocessing_fnc), A.Lambda(image=to_tensor, mask=to_tensor)])
    return _transform


# ------------------- DATASET CREATION AND DATA VISUALIZATION -------------------
seq = '00001'

# datasets
dataset = Dataset(seq=f'rellis_3d/{seq}', classes=['grass'])
augmented_dataset = Dataset(seq=f'rellis_3d/{seq}', augmentation=get_training_augmentation(), classes=['grass'])

# original image
image, mask = dataset[1]
visualize(image=image[..., (2, 1, 0)], grass_mask=mask.squeeze())

# same image with different random transforms
# generates new image everytime is item called
for i in range(3):
    image, mask = augmented_dataset[1]
    visualize(image=image[..., (2, 1, 0)], mask=mask.squeeze(-1))

# ------------------- MODEL CREATION -------------------

ENCODER = "se_resnext50_32x4d"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["grass"]
ACTIVATION = "sigmoid"
DEVICE = "cuda"

# Create model and preprocessing function
model = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES), activation=ACTIVATION)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(seq=f'rellis_3d/{seq}',
                        augmentation=get_training_augmentation(),
                        preprocessing=get_preprocessing(preprocessing_fn),
                        classes=CLASSES)

valid_dataset = Dataset(seq=f'rellis_3d/{seq}',
                        augmentation=get_validation_augmentation(),
                        preprocessing=get_preprocessing(preprocessing_fn),
                        classes=CLASSES)

# ------------------- MODEL TRAINING -------------------

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# Create loss function, metric and optimization algorithm
loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=DEVICE,
                                         verbose=True)

valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True)

# train model for 40 epochs
max_score = 0
# for i in range(0, 40):
for i in range(0, 2):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# ## Test best saved model
# load best saved checkpoint
best_model = torch.load('./best_model.pth')

# ## Visualize predictions
# test dataset without transformations for image visualization
test_dataset = valid_dataset

for i in range(5):
    n = np.random.choice(len(test_dataset))

    image_vis = test_dataset[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    visualize(ground_truth_mask=gt_mask, predicted_mask=pr_mask)
