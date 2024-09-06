import os
import torch
import matplotlib.pyplot as plt
from monai.bundle import ConfigParser
from monai.transforms import LoadImage, EnsureChannelFirst, Compose
import logging

# Set up logging
logging.basicConfig(filename='segmentation_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

scans_folder = "D:\\Datasets\\FDG-PET-CT-Lesions\\NIFTI\\FDG-PET-CT-Lesions"
whole_body_ct_segmentation_folder = "D:\\Datasets\\FDG-PET-CT-Lesions\\Whole Body CT Segmentation"
whole_body_ct_segmentation_metadata_path = os.path.join(whole_body_ct_segmentation_folder, "configs\metadata.json")

patient_scan_paths = []
for root_directory, patient_directories, _ in os.walk(scans_folder):
    for patient_directory in patient_directories:
        patient_path = os.path.join(root_directory, patient_directory)
        for _, patient_scan_dates, _ in os.walk(patient_path):
            for patient_scan_date in patient_scan_dates:
                patient_scan_paths.append(os.path.join(patient_path, patient_scan_date))

# Process each patient scan path separately
for patient_scan_path in patient_scan_paths:
    try:
        if (os.path.isfile(os.path.join(patient_scan_path, 'CTres.nii.gz')) and not os.path.isdir(os.path.join(patient_scan_path, 'mask'))):
            logging.info(f"Creating organ segmentation mask for patient: {patient_scan_path}")
            
            organ_segmentation_mask_path = os.path.join(patient_scan_path, 'mask')
            CTRes = os.path.join(patient_scan_path, "CTres.nii.gz")

            # Lets now try segment all the organs of the whole body scan
            model_name = "wholeBody_ct_segmentation"
            #download(name=model_name, bundle_dir=whole_body_ct_segmentation_folder)

            # Now specify the model path (where the actual model parameters are saved) in this case we're using the lowres model
            model_path = os.path.join(whole_body_ct_segmentation_folder, "wholeBody_ct_segmentation", "models", "model_lowres.pt")
            config_path = os.path.join(whole_body_ct_segmentation_folder, "wholeBody_ct_segmentation", "configs", "inference.json")

            config = ConfigParser()
            config.read_config(config_path)
            
            # Dynamically update the output_dir
            config["output_dir"] = organ_segmentation_mask_path
            
            preprocessing = config.get_parsed_content("preprocessing")
            data = preprocessing({'image': CTRes})

            model = config.get_parsed_content("network")

            # Explicitly set the device
            device = torch.device('cpu')

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            inferer = config.get_parsed_content("inferer")
            postprocessing = config.get_parsed_content("postprocessing")
            data = preprocessing({"image": CTRes})
            with torch.no_grad():
                data['image'] = data['image'].to(device)  # Ensure image is on the correct device
                data['pred'] = inferer(data['image'].unsqueeze(0), network=model)

            data['pred'] = data['pred'][0]
            data['image'] = data['image'][0]

            data = postprocessing(data)
            #segmentation = torch.flip(data['pred'][0], dims=[2])
            segmentation = data['pred'][0]
            segmentation = segmentation.cpu().numpy()
            
            preprocessing_pipeline = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
            ])
            
            logging.info(f"Successfully processed segmentation for patient at: {organ_segmentation_mask_path}")
    except Exception as e:
        logging.error(f"Error processing patient scan: {patient_scan_path}. Exception: {str(e)}")

"""
    # And then we can open using this preprocessing pipeline
    CT = preprocessing_pipeline(CTRes)
    
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
"""