import os, fnmatch
import nibabel as nib

# Taken from https://stackoverflow.com/questions/2186525/how-to-use-to-find-files-recursively
def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results

def load_ctres(file):
    return nib.load(file).get_fdata()

data_dir = "D:\\Datasets\\FDG-PET-CT-Lesions\\NIFTI\\FDG-PET-CT-Lesions"
train_images = sorted(recursive_glob(data_dir, '*CTres_lung_region.nii.gz'))

minimum_width = 0
minimum_height = 0
minimum_slice_count = 0
maximum_width = 0
maximum_height = 0
maximum_slice_count = 0
counter = 0
for train_image in train_images:
    ctres_file = load_ctres(train_image)
    
    try:
        if minimum_width == 0 or ctres_file.shape[0] < minimum_width:
            minimum_width = ctres_file.shape[0]
        if minimum_height == 0 or ctres_file.shape[1] < minimum_height:
            minimum_height = ctres_file.shape[1]
        if minimum_slice_count == 0 or ctres_file.shape[2] < minimum_slice_count:
            minimum_slice_count = ctres_file.shape[2]
        if maximum_width == 0 or ctres_file.shape[0] > maximum_width:
            maximum_width = ctres_file.shape[0]
        if maximum_height == 0 or ctres_file.shape[1] > maximum_height:
            maximum_height = ctres_file.shape[1]
        if maximum_slice_count == 0 or ctres_file.shape[2] > maximum_slice_count:
            maximum_slice_count = ctres_file.shape[2]
            
    except Exception as e:
        print(f"An error occurred trying to parse image: {train_image}")
        
    if counter % 100 == 0:
        print("100 images processed")
        
    counter = counter + 1
        
print(f"Minimum width, height and slices: {minimum_width}, {minimum_height}, {minimum_slice_count}")
print(f"Maximum width, height and slices: {maximum_width}, {maximum_height}, {maximum_slice_count}")