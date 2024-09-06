import os, fnmatch
from glob import glob
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstD,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized
)
from monai.data import Dataset, DataLoader
from monai.utils import first
import matplotlib.pyplot as plt

# Taken from https://stackoverflow.com/questions/2186525/how-to-use-to-find-files-recursively
def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results

data_dir = "D:\\Datasets\\FDG-PET-CT-Lesions\\Tutorials"
train_images = sorted(recursive_glob(os.path.join(data_dir, 'TrainData'), '*CTres.nii.gz'))
train_labels = sorted(recursive_glob(os.path.join(data_dir, 'TrainLabels'), '*.nii.gz'))

val_images = sorted(recursive_glob(os.path.join(data_dir, 'ValData'), '*CTres.nii.gz'))
val_lables = sorted(recursive_glob(os.path.join(data_dir, 'ValLabels'), '*.nii.gz'))

train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
val_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(val_images, val_lables)]

orig_transforms = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstD(keys=['image', 'label']),
        ToTensord(keys=['image', 'label']) # ToTensor always needs to come last, as it needs to be the final transform operation before it gets converted to a tensor
    ]
)

train_transforms = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstD(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),
        ScaleIntensityRanged(keys=['image', 'label'], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        Resized(keys=['image', 'label'], spatial_size=[128, 128, 128]),
        ToTensord(keys=['image', 'label'])
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstD(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),
        ScaleIntensityRanged(keys=['image', 'label'], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=['image', 'label'])
    ]
)

orig_ds = Dataset(data=train_files, transform=orig_transforms)
orig_loader = DataLoader(orig_ds, batch_size=1)

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

test_patient = first(train_loader)
orig_patient = first(orig_loader)

plt.figure('test', (12, 6))

plt.subplot(1, 3, 1)
plt.title('Orig Patient')

# ['image'][batch_size, channel, width, height, slice]
plt.imshow(orig_patient['image'][0, 0, :, :, 100], cmap="gray")

plt.subplot(1, 3, 2)
plt.title('Slice of a Patient')
plt.imshow(test_patient['image'][0, 0, :, :, 100], cmap="gray")

plt.subplot(1, 3, 3)
plt.title('Label of a Patient')
plt.imshow(test_patient['label'][0, 0, :, :, 100], cmap="gray")
plt.show()