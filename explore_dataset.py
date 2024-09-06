import h5py
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tkinter as tk
from tkinter import ttk, Checkbutton

# Open the HDF5 file
file_path = 'D:\\Datasets\\FDG-PET-CT-Lesions\\HDF5\\FDG-PET-CT-Lesions.hdf5'

# Recursively explore the file
def explore_group(group, level=0):
    indent = "  " * level
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"{indent}Group: {key}")
            explore_group(item, level + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}Dataset: {key}, Shape: {item.shape}, Data type: {item.dtype}")
            
            # Load CT image
            if key == "ct":
                ct_images = np.array(item)
                 
                # Example: Display the first image slice using matplotlib
                # plt.imshow(ct_images[:, :, 0], cmap='gray')
                # plt.show()
                
            # Load CTRES image
            if key == "ctres":
                ctres_images = np.array(item)
                
                # Example: Display the first image slice using matplotlib
                plt.imshow(ctres_images[:, :, 0], cmap='gray')
                plt.show()
                
            # Load PET image
            if key == "pet":
                pet_images = np.array(item)
                
                # Example: Display the first image slice using matplotlib
                # plt.imshow(pet_images[:, :, 0], cmap='gray')
                # plt.show()
                
            # Load SEG image
            if key == "seg":
                seg_images = np.array(item)
                
                # Example: Display the first image slice using matplotlib
                # plt.imshow(seg_images[:, :, 0], cmap='gray')
                # plt.show()
                
            # Load SUV image
            if key == "suv":
                suv_images = np.array(item)
                
                # Example: Display the first image slice using matplotlib
                # plt.imshow(suv_images[:, :, 0], cmap='gray')
                # plt.show()


with h5py.File(file_path, 'r') as hdf:
    for group in hdf.keys():
        print(group)

    # Start exploring from the root group
    explore_group(hdf)