import os
import fnmatch
import nibabel as nib
import numpy as np
import logging

# Set up logging
logging.basicConfig(filename='lung_isolator_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to find files recursively
def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results

# Function to save the lung slices and other relevant organs
def save_lung_related_slices(input_file, output_file, lung_related_mask):
    img = nib.load(input_file)
    img_data = img.get_fdata()

    # Apply lung-related mask to extract slices containing lungs or related organs
    relevant_slices = img_data[:, :, np.any(lung_related_mask, axis=(0, 1))]

    # Create a new NIfTI image with the relevant slices
    relevant_img = nib.Nifti1Image(relevant_slices, img.affine)
    nib.save(relevant_img, output_file)

def main(data_dir):
    # Labels for lungs and lung related organs
    lung_related_labels = [13, 14, 15, 16, 17, 42, 43, 44, 45, 46, 47, 48, 49]  # Lungs, trachea, esophagus, heart, pulmonary artery
    
    ct_scan_files = sorted(recursive_glob(data_dir, '*CTres.nii.gz'))
    pet_scan_files = sorted(recursive_glob(data_dir, '*SUV.nii.gz'))
    cancer_segmentation_files = sorted(recursive_glob(data_dir, '*SEG.nii.gz'))
    organ_segmentation_files = sorted(recursive_glob(data_dir, '*CTres_trans.nii.gz'))

    # Iterate over all scans
    for ct_scan_file, pet_scan_file, cancer_segmentation_file, organ_segmentation_file in zip(ct_scan_files, pet_scan_files, cancer_segmentation_files, organ_segmentation_files):
        patient_id = ct_scan_file.split("\\")[5]
        ct_scan_file_lung_region_file = (os.path.dirname(ct_scan_file) + "\\" + os.path.basename(ct_scan_file).replace('.nii.gz', '_lung_region.nii.gz'))
        pet_scan_file_lung_region_file = (os.path.dirname(pet_scan_file) + "\\" + os.path.basename(pet_scan_file).replace('.nii.gz', '_lung_region.nii.gz'))
        cancer_segmentation_file_lung_region_file = (os.path.dirname(cancer_segmentation_file) + "\\" + os.path.basename(cancer_segmentation_file).replace('.nii.gz', '_lung_region.nii.gz'))
        organ_segmentation_file_lung_region_file = (os.path.dirname(organ_segmentation_file) + "\\" +  os.path.basename(organ_segmentation_file).replace('.nii.gz', '_lung_region.nii.gz'))
        
        # Iterate over all scans
        if  (not os.path.isfile(ct_scan_file_lung_region_file) or not os.path.isfile(pet_scan_file_lung_region_file) or not os.path.isfile(cancer_segmentation_file_lung_region_file) or not os.path.isfile(organ_segmentation_file_lung_region_file)):
            logging.info(f"Isolating lung region for scans of: {patient_id}")
            
            # Load the organ segmentation to identify relevant slices
            organ_img = nib.load(organ_segmentation_file)
            organ_data = organ_img.get_fdata()

            # Create a mask for lung-related structures
            lung_region_mask = np.isin(organ_data, lung_related_labels)

            # Save the lung-related slices for each type of scan
            save_lung_related_slices(ct_scan_file, ct_scan_file_lung_region_file, lung_region_mask)
            save_lung_related_slices(pet_scan_file, pet_scan_file_lung_region_file, lung_region_mask)
            save_lung_related_slices(cancer_segmentation_file, cancer_segmentation_file_lung_region_file, lung_region_mask)
            save_lung_related_slices(organ_segmentation_file, organ_segmentation_file_lung_region_file, lung_region_mask)

    logging.info("All files have been processed")

# Execute the main function
if __name__ == "__main__":
    data_dir = "D:\\Datasets\\FDG-PET-CT-Lesions\\NIFTI\\FDG-PET-CT-Lesions"
    main(data_dir)