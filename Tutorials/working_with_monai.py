import os
import numpy
import torch
import pydicom
import matplotlib.pyplot as plt
from monai.bundle import ConfigParser, download
from monai.transforms import LoadImage, LoadImage, Orientation, Orientationd, EnsureChannelFirst, EnsureChannelFirstd, Compose, Spacingd, Affined
from rt_utils import RTStructBuilder
import json

scans_folder = "D:\\Datasets\\FDG-PET-CT-Lesions\\NIFTI\\FDG-PET-CT-Lesions"
whole_body_ct_segmentation_folder = "D:\\Datasets\\FDG-PET-CT-Lesions\\Whole Body CT Segmentation"
patient_0af7ffe12a_CT_folder = os.path.join(scans_folder, "PETCT_0af7ffe12a\\08-12-2005-NA-PET-CT Ganzkoerper  primaer mit KM-96698\\CTres.nii.gz")
patient_0af7ffe12a_PET_folder = os.path.join(scans_folder, "PETCT_0af7ffe12a\\08-12-2005-NA-PET-CT Ganzkoerper  primaer mit KM-96698\\3.000000-PET corr.-88124")

# Each dicom file is a separate image, an axial \ 2d slice from the top down perspective. 
# Stored as separate files because if stored together, it would be a much larger file size, which on older systems would have been
# a challenge. Dicom is an old standard so this is less of a issue today with more modern systems.
"""
# Read in the 232nd axial slice of patient 0af7ffe12a's CT scan. This function will return a pydicom dataset
ds = pydicom.read_file(os.path.join(patient_0af7ffe12a_CT_folder, "1-232.dcm"))

# We can find all of the data a physician would need about the CT scan by printing the dicom dataset's metadata 
print(ds)

# The image data of the pydicom dataset is contained within the pixel_array property
image = ds.pixel_array

# To find the image resolution we can print image.shape
print(image.shape)

# The image is a 2D array. For CT scans, specific units are used known as Hounsfield units. But when the images are saved as dicom,
# sometimes the units are converted to a format that helps them be converted to intergers to they can be saved easier (to make it 
# more convenient to store the image)
# To change the image back to use Hounsfield units, we can use the RescaleSlope and RecaleIntercept functions of the dicom dataset . 
# Hounsfield units can be thought of as measuring density, -1000 corresponds to air,  0 corresponds to water, 1000+ corresponds to 
# bone.
image = ds.RescaleSlope * image + ds.RescaleIntercept

# We can now plot the individual image slice
plt.pcolormesh(image, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')
plt.show()

# We can use the Medical Open Network for Artificial Intelligence extensdion of pytorch for machine learning with medical data, it
# contains many important functions for manipulating medical data. 
# MONAI has functionality for easily opening medical data
# Lets open the entire CT scan folder of patient 0af7ffe12a using MONAI's LoadImage functionality. We'll first define an instance
# of the LoadImage class an assign it to image_loader variable
image_loader = LoadImage(image_only=True)
CT = image_loader(patient_0af7ffe12a_CT_folder)

# If we print the CT image, it will output a metatensor structure, which for all intents and purposes works the same as a pytorch
# metatensor, but the main difference is that it has a meta attribute, which is very useful for medical imaging. Metatensors provide
# things like the pixel spacing (the milimeter resolution in the axial plane and the resolution in the z slices), the position of 
# the patient, the shape of the array, the affine matrices (used to line up position of the patient across 2 different scans)
print(CT)
print(CT.meta)

# We don't need to just display the single top down 2D slice, we can display the full coronal (a front slice) or sagital (a slice
# cut in the center) 3D slices 

# Ensure CT is a 3D array and select the 256th coronal slice
CT_coronal_slice = CT[:, 256]  # Selecting the middle slice along the third dimension

# Plot the coronal slice
plt.figure(figsize=(3, 8))
plt.pcolormesh(CT_coronal_slice.T, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')

#CT_coronal_slice = CT_coronal_slice.cpu().numpy()

# Prints the coronal figure is upside down
plt.show()

# The issue with this coronal slice is that it displays upside down. The patient is not oriented properly.
# To get around this, we can use a MONAI transform. 

# First  we need to find the current shape of the images. This gives a [512,512,975] 3D array. Typically there is a channel dimension 
# and a batch dimension when dealing with data. As we can see from printing this CT.shape, there is currently no channel dimension
# definied. The channel dimension is important because when dealing with colour images, you need 3 different parameters to tell you 
# the colour of each individual voxel. In our case, we are dealing with density only (1 number tells you everything you need to know
# about a particular voxel's location in space) so we just need a 1 dimensional channel. 
# a 1D channel
print(CT.shape)

# Now that we know the shape of the images, we can add the 1D channel transform. 
channel_transform = EnsureChannelFirst()
CT = channel_transform(CT)

# Looking at the shape of the images now, it now contains the channel dimension as the first element of the shape of the images 
# [1,512,512,975]
print(CT.shape)

# Now we'll want to create a orientation transform so that we can re-orient the image
# Orientation corresponds to x, y and z orientation, L stands for Left, P stands for Posterior and S stands for Superior.
# The transform orients the image in such a way that when we load the image, and look at a particular coronial slice, we
# get the image oriented exactly as we expect. 
orientation_transform = Orientation(axcodes=("LPS"))
CT = orientation_transform(CT)

# Now obtain the coronial slice
CT_coronal_slice = CT[0,:,256].cpu().numpy()

# Now we can plot again with the correct orientation set (due to setting axcodes to LPS)
plt.figure(figsize=(5, 10))
plt.pcolormesh(CT_coronal_slice.T, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')
plt.show()

"""

# Alternatively, we can combine all these transforms in one go when we initially open the image data by using a pipeline
preprocessing_pipeline = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes='LPS')
])

# And then we can open using this preprocessing pipeline
CT = preprocessing_pipeline(patient_0af7ffe12a_CT_folder)
CT_coronal_slice = CT[0,:,200].cpu().numpy()

# And plot
plt.figure(figsize=(5,10))
plt.pcolormesh(CT_coronal_slice.T, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')
plt.show()

"""
# Given we have CT and PET scans, we want to be able to have them both correspond to the same transform structure.
# The transform to use for this is called a dictionary transform. By adding a d to the end of the transforms we make them dictionary 
# transform. Unlike LoadImage which just takes in a folder path of the images, LoadImaged takes in a dictionary. So we specify a 
# ct_image key which maps to the folder path of the CT images, and then a pet_image key which maps to the folder path of the pet 
# images
data = {'ct_image': patient_0af7ffe12a_CT_folder, 'pet_image': patient_0af7ffe12a_PET_folder}

# We add Spacingd to resample images to the same spacing
# and Affined for an affine transformation that aligns PET to CT.
preprocessing_pipeline = Compose([
    LoadImaged(keys=['ct_image','pet_image'], image_only=True),
    EnsureChannelFirstd(keys=['ct_image','pet_image']),
    Orientationd(keys=['ct_image','pet_image'], axcodes='LPS'),
    Spacingd(keys=['ct_image', 'pet_image'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'bilinear')),
    Affined(keys=['pet_image'], mode='bilinear', padding_mode='border', spatial_size=CT.shape[1:])
])

data = preprocessing_pipeline(data)

# Extract the CT and PET images after preprocessing
CT = data['ct_image']
PET = data['pet_image']

# Extract the same 3D slice from both CT and PET scans
CT_coronal_slice = CT[0, :, 256].cpu().numpy()
PET_coronal_slice = PET[0, :, 256].cpu().numpy()

# Plotting the CT coronal slice
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.pcolormesh(CT_coronal_slice.T, cmap='Greys_r')
plt.colorbar(label='HU')
plt.title('CT Coronal Slice')
plt.axis('off')

# Plotting the PET coronal slice
plt.subplot(1, 2, 2)
plt.pcolormesh(PET_coronal_slice.T, cmap='hot')
plt.colorbar(label='PET Intensity')
plt.title('PET Coronal Slice')
plt.axis('off')

plt.show()

"""

# Lets now try segment all the organs of the whole body scan
model_name = "wholeBody_ct_segmentation"
#download(name=model_name, bundle_dir=whole_body_ct_segmentation_folder)

# Now specify the model path (where the actual model parameters are saved) in this case we're using the lowres model
model_path = os.path.join(whole_body_ct_segmentation_folder, "wholeBody_ct_segmentation", "models", "model_lowres.pt")
config_path = os.path.join(whole_body_ct_segmentation_folder, "wholeBody_ct_segmentation", "configs", "inference.json")

config = ConfigParser()
config.read_config(config_path)
config["output_dir"] = os.path.join(scans_folder, "PETCT_0af7ffe12a\\08-12-2005-NA-PET-CT Ganzkoerper  primaer mit KM-96698")

preprocessing = config.get_parsed_content("preprocessing")
data = preprocessing({'image': patient_0af7ffe12a_CT_folder})
print(data)

model = config.get_parsed_content("network")

# Explicitly set the device
device = torch.device('cpu')

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

inferer = config.get_parsed_content("inferer")
postprocessing = config.get_parsed_content("postprocessing")

#datalist = [patient_0af7ffe12a_CT_folder]
#config["datalist"] = datalist
#dataloader = config.get_parsed_content["dataloader"]

data = preprocessing({"image": patient_0af7ffe12a_CT_folder})
with torch.no_grad():
    data['image'] = data['image'].to(device)  # Ensure image is on the correct device
    data['pred'] = inferer(data['image'].unsqueeze(0), network=model)
    
data['pred'] = data['pred'][0]
data['image'] = data['image'][0]

data = postprocessing(data)
segmentation = torch.flip(data['pred'][0], dims=[2])
segmentation = segmentation.cpu().numpy()

slice_idx = 200
CT_coronal_slice = CT[0,:,slice_idx].cpu().numpy()
segmentation_coronal_slice = segmentation[:,slice_idx]

plt.subplots(1,2,figsize=(6,8))
plt.subplot(121)
plt.pcolormesh(CT_coronal_slice.T, cmap='Greys_r')
plt.axis('off')
plt.subplot(122)
plt.pcolormesh(segmentation_coronal_slice.T, cmap='nipy_spectral')
plt.axis('off')
plt.show()
