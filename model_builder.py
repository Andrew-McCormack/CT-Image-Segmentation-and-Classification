import os
import fnmatch
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstD,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    ResizeWithPadOrCropd,
    RandRotate90d,
    RandFlipd,
    RandAffined,
    NormalizeIntensityd
)
from monai.data import Dataset, DataLoader
from monai.utils import first
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename='model_builder_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def get_model():
    return SegResNet(
        spatial_dims=3,
        in_channels=3,  # We combine CT, PET, and organ segmentation into a single 3-channel input
        out_channels=1,  # Single channel output for cancer segmentation
        init_filters=16,
        #init_filters=8,
        dropout_prob=0.2,
        norm="instance",
        act="leakyrelu",
    )

# Function to find files recursively
def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results

# Function to load and split the dataset
def load_and_split_data(data_dir, csv_path, test_size=0.25, random_state=104):
    # Load the CSV file
    clinical_data = pd.read_csv(csv_path)
    
    # Filter the rows where the Diagnosis is either 'NEGATIVE' or 'LUNG_CANCER'
    filtered_data = clinical_data[clinical_data['diagnosis'].isin(['NEGATIVE', 'LUNG_CANCER'])]
    
    # Extract the Subject IDs that meet the condition
    subject_ids = filtered_data['Subject ID'].astype(str).tolist()
    
    # Find all the relevant files and filter them based on Subject IDs
    ct_scans = sorted([f for f in recursive_glob(data_dir, '*CTres_lung_region.nii.gz') if any(sub_id in f for sub_id in subject_ids)])
    pet_scans = sorted([f for f in recursive_glob(data_dir, '*SUV_lung_region.nii.gz') if any(sub_id in f for sub_id in subject_ids)])
    cancer_segmentations = sorted([f for f in recursive_glob(data_dir, '*SEG_lung_region.nii.gz') if any(sub_id in f for sub_id in subject_ids)])
    organ_segmentations = sorted([f for f in recursive_glob(data_dir, '*CTres_trans_lung_region.nii.gz') if any(sub_id in f for sub_id in subject_ids)])
    
    return train_test_split(
        ct_scans, pet_scans, cancer_segmentations, organ_segmentations, 
        random_state=random_state, test_size=test_size, shuffle=True
    )

# Function to create data dictionaries for MONAI
def create_data_dicts(ct_scans, pet_scans, organ_segmentations, cancer_segmentations):
    return [
        {"ct_scan": ct_scan, "pet_scan": pet_scan, "organ_segmentation": organ_segmentation, "cancer_segmentation": cancer_segmentation}
        for ct_scan, pet_scan, organ_segmentation, cancer_segmentation in zip(ct_scans, pet_scans, organ_segmentations, cancer_segmentations)
    ]

# Function to define transforms
def get_transforms(mode="train"):
    if mode == "train":
        return Compose([
            # Cropping is always preferred over reducing detail. Look into using Google Colab to allow for concurrent testing of different models \ params
            LoadImaged(keys=['ct_scan', 'pet_scan', 'organ_segmentation', 'cancer_segmentation']),
            EnsureChannelFirstD(keys=['ct_scan', 'pet_scan', 'organ_segmentation', 'cancer_segmentation']),
            Spacingd(keys=['ct_scan', 'pet_scan', 'organ_segmentation', 'cancer_segmentation'], pixdim=(1.0, 1.0, 1.0)),
            CropForegroundd(keys=['ct_scan', 'pet_scan'], source_key='ct_scan'),
            ResizeWithPadOrCropd(keys=['ct_scan', 'pet_scan', 'organ_segmentation', 'cancer_segmentation'], spatial_size=[96, 192, 192]),
            NormalizeIntensityd(keys=['ct_scan', 'pet_scan'], channel_wise=True),
            ToTensord(keys=['ct_scan', 'pet_scan', 'organ_segmentation', 'cancer_segmentation'])
        ])
    elif mode == "validation":
        return Compose([
            LoadImaged(keys=['ct_scan', 'pet_scan', 'organ_segmentation', 'cancer_segmentation']),
            EnsureChannelFirstD(keys=['ct_scan', 'pet_scan', 'organ_segmentation', 'cancer_segmentation']),
            Spacingd(keys=['ct_scan', 'pet_scan', 'organ_segmentation', 'cancer_segmentation'], pixdim=(1.0, 1.0, 1.0)),
            CropForegroundd(keys=['ct_scan', 'pet_scan'], source_key='ct_scan'),
            ResizeWithPadOrCropd(keys=['ct_scan', 'pet_scan', 'organ_segmentation', 'cancer_segmentation'], spatial_size=[96, 192, 192]),
            NormalizeIntensityd(keys=['ct_scan', 'pet_scan'], channel_wise=True),
            ToTensord(keys=['ct_scan', 'pet_scan', 'organ_segmentation', 'cancer_segmentation'])
        ])

# Function to create datasets and dataloaders
def create_datasets_and_loaders(train_files, validation_files, batch_size=2):
    train_ds = Dataset(data=train_files, transform=get_transforms("train"))
    #train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=8, pin_memory=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=6, pin_memory=True, prefetch_factor=2)

    val_ds = Dataset(data=validation_files, transform=get_transforms("validation"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=6, pin_memory=True, prefetch_factor=2)

    return train_loader, val_loader

# Function to visualize a batch of data
def visualize_batch(loader, slice_idx=60):
    patient = first(loader)

    plt.figure('test', (12, 6))
    plt.subplot(1, 3, 1)
    plt.title('CT Image')
    plt.imshow(patient['ct_scan'][0, 0, :, :, slice_idx], cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title('PET Image')
    plt.imshow(patient['pet_scan'][0, 0, :, :, slice_idx], cmap="hot")

    plt.subplot(1, 3, 3)
    plt.title('Cancer Segmentation Label')
    plt.imshow(patient['cancer_segmentation'][0, 0, :, :, slice_idx], cmap="gray")

    plt.show()

# Training loop with mixed precision and optimization for memory usage
def train_model(train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    # Loss function, optimizer, and scaler for mixed precision
    loss_function = DiceLoss(sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_counter = 1 
        for batch_data in train_loader:
            logging.info(f"Batch {batch_counter}")
            ct_scan = batch_data["ct_scan"].to(device)
            pet_scan = batch_data["pet_scan"].to(device)
            organ_segmentation = batch_data["organ_segmentation"].to(device)
            cancer_segmentation = batch_data["cancer_segmentation"].to(device)

            # Combine inputs along the channel dimension
            inputs = torch.cat((ct_scan, pet_scan, organ_segmentation), dim=1)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, cancer_segmentation)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batch_counter = batch_counter + 1

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            dice_scores = []
            for val_data in val_loader:
                val_ct = val_data["ct_scan"].to(device)
                val_pet = val_data["pet_scan"].to(device)
                val_organ = val_data["organ_segmentation"].to(device)
                val_label = val_data["cancer_segmentation"].to(device)

                val_inputs = torch.cat((val_ct, val_pet, val_organ), dim=1)

                with autocast():
                    val_outputs = model(val_inputs)
                    dice_scores.append(dice_metric(val_outputs, val_label).item())
            
            logging.info(f"Validation Dice Score: {sum(dice_scores) / len(dice_scores)}")

# Main function to execute the workflow
def main(data_dir, csv_path):
    # Load and split the data
    train_ct_scans, validation_ct_scans, train_pet_scans, validation_pet_scans, \
    train_cancer_segmentations, validation_cancer_segmentations, \
    train_organ_segmentations, validation_organ_segmentations = load_and_split_data(data_dir, csv_path)

    # Create data dictionaries
    train_files = create_data_dicts(train_ct_scans, train_pet_scans, train_organ_segmentations, train_cancer_segmentations)
    validation_files = create_data_dicts(validation_ct_scans, validation_pet_scans, validation_organ_segmentations, validation_cancer_segmentations)

    # Create datasets and loaders
    train_loader, validation_loader = create_datasets_and_loaders(train_files, validation_files, batch_size=2)

    #visualize_batch(train_loader, 198)

    # Train the model
    train_model(train_loader, validation_loader)

# Execute the main function
if __name__ == "__main__":
    csv_path = "D:\\Datasets\\FDG-PET-CT-Lesions\\Clinical-Metadata-FDG-PET_CT-Lesions.csv"
    data_dir = "D:\\Datasets\\FDG-PET-CT-Lesions\\NIFTI\\FDG-PET-CT-Lesions"
    main(data_dir, csv_path)