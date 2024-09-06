import os
import numpy as np
import nibabel as nib
import tkinter as tk
import matplotlib.pyplot as plt
import json
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.cm as cm

base_path = 'D:\\Datasets\\FDG-PET-CT-Lesions\\nifti\\FDG-PET-CT-Lesions'
ORGAN_LABELS = []
patients = {}
patient_var = ""
scan_date_var = ""
scan_type_var = ""
scan_region_var = ""

class NiftiLoader:
    def __init__(self, base_path, patient_id, scan_date, scan_region):
        if (scan_region == "Lungs"):
            self.ctres_file = os.path.join(base_path, patient_id, scan_date, "CTres_lung_region.nii.gz")
            self.cancer_segmentation_file = os.path.join(base_path, patient_id, scan_date, "SEG_lung_region.nii.gz")
            self.suv_file = os.path.join(base_path, patient_id, scan_date, "SUV_lung_region.nii.gz")
            self.organ_segmentation_file = os.path.join(base_path, patient_id, scan_date, "mask\\CTres\\CTres_trans_lung_region.nii.gz")
        else:
            self.ctres_file = os.path.join(base_path, patient_id, scan_date, "CTres.nii.gz")
            self.cancer_segmentation_file = os.path.join(base_path, patient_id, scan_date, "SEG.nii.gz")
            self.suv_file = os.path.join(base_path, patient_id, scan_date, "SUV.nii.gz")
            self.organ_segmentation_file = os.path.join(base_path, patient_id, scan_date, "mask\\CTres\\CTres_trans.nii.gz")
    
    def load_ctres(self):
        return nib.load(self.ctres_file).get_fdata()
    
    def load_cancer_segmentation(self):
        return nib.load(self.cancer_segmentation_file).get_fdata()
    
    def load_suv(self):
        return nib.load(self.suv_file).get_fdata()

    def load_organ_segmentation(self):
        return nib.load(self.organ_segmentation_file).get_fdata()

class ImageViewer:
    def __init__(self, master, image_files, image_type, suv_overlay, organ_segmentation_overlay, cancer_segmentation_data=None, organ_segmentation_data=None):
        self.master = master
        self.image_files = image_files
        self.suv_overlay = suv_overlay
        self.organ_segmentation_overlay = organ_segmentation_overlay
        self.cancer_segmentation_data = cancer_segmentation_data
        self.organ_segmentation_data = organ_segmentation_data
        self.view = image_type.split()[0]  # Extract 'Axial', 'Coronal', or 'Sagittal' from image_type

        self.start_slice = 0
        self.end_slice = len(self.image_files) 
        
        self.max_slices = self.end_slice - self.start_slice
        self.current_image_index = 0  # Start from the first valid slice
        
        self.master.title(image_type + " Viewer")
        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.create_scrollable_legend()
        self.update_image()

        self.master.bind("<MouseWheel>", self.scroll_images)
        self.master.bind("<Button-4>", self.scroll_images)  # For Linux systems
        self.master.bind("<Button-5>", self.scroll_images)  # For Linux systems

    def create_scrollable_legend(self):
        # Legend Frame
        self.legend_frame = tk.Frame(self.master)
        self.legend_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a canvas widget
        self.canvas = tk.Canvas(self.legend_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a vertical scrollbar to the canvas
        self.scrollbar = tk.Scrollbar(self.legend_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas to use the scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create a frame within the canvas to hold the legend items
        self.legend_inner_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.legend_inner_frame, anchor="nw")

        # Populate the legend
        self.populate_legend()

        # Configure scrolling region
        self.legend_inner_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def populate_legend(self):
        colormap = plt.cm.get_cmap('nipy_spectral', len(ORGAN_LABELS) + 1)

        for i, (label, name) in enumerate(ORGAN_LABELS.items()):
            if name == 'Cancer':
                hex_color = '#FF6EC7'  # Pink color for Cancer
            else:
                color = colormap(i / len(ORGAN_LABELS))
                color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                hex_color = f'#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}'

            # Frame for each legend item
            item_frame = tk.Frame(self.legend_inner_frame)
            item_frame.pack(fill=tk.X, padx=5, pady=2)

            # Label displaying the organ name
            label_widget = tk.Label(item_frame, text=name, anchor="w", width=20)
            label_widget.pack(side=tk.LEFT)

            # Label displaying the color block
            color_block = tk.Label(item_frame, bg=hex_color, width=2)
            color_block.pack(side=tk.LEFT, padx=(5, 0))

    def scroll_images(self, event):
        if event.delta > 0 or event.num == 4:
            self.current_image_index = min(self.max_slices - 1, self.current_image_index + 1)
        elif event.delta < 0 or event.num == 5:
            self.current_image_index = max(0, self.current_image_index - 1)
        self.update_image()

    def apply_colourmap(self, image_array, colormap_name='nipy_spectral'):
        # Create a base RGB image for the organ segmentation
        colormap = plt.cm.get_cmap(colormap_name, len(ORGAN_LABELS))
        label_to_color = np.array([(0, 0, 0)] + [colormap(label / len(ORGAN_LABELS))[:3] for label in ORGAN_LABELS.keys()]) * 255
        label_to_color = label_to_color.astype(np.uint8)

        image_array = image_array.astype(int)
        colored_image = label_to_color[image_array]

        return colored_image

    def overlay_images(self, base_image, organ_segmentation=None, cancer_segmentation=None, alpha=0.5):
        if organ_segmentation is not None:
            organ_colored = self.apply_colourmap(organ_segmentation, colormap_name='nipy_spectral')

            # Resize organ_colored to match the base_image size
            organ_colored = Image.fromarray(organ_colored)
            organ_colored = organ_colored.resize((base_image.shape[1], base_image.shape[0]))
            organ_colored = np.array(organ_colored)

            # Combine base image and organ segmentation
            base_image = (1 - alpha) * base_image + alpha * organ_colored
            base_image = np.clip(base_image, 0, 255).astype(np.uint8)

        if cancer_segmentation is not None:
            cancer_color = np.array([255, 110, 199], dtype=np.uint8)  # Pink color for cancer
            cancer_overlay = np.zeros_like(base_image)

            # Apply pink to the areas where the cancer segmentation is active
            cancer_overlay[cancer_segmentation > 0] = cancer_color

            # Blend the cancer overlay with the existing base image
            base_image = np.where(cancer_overlay > 0, cancer_overlay, base_image)

        return base_image

    def update_image(self):
        # Adjust the actual slice index based on the view
        actual_slice_index = self.current_image_index + self.start_slice

        print(f"Displaying image {self.current_image_index + 1} of {self.max_slices} (Actual Slice: {actual_slice_index + 1})")
        image_array = self.image_files[actual_slice_index]
        print(f"Image shape: {image_array.shape}, dtype: {image_array.dtype}")

        # Flip the slices as needed to correct the orientation
        if self.view == "Axial":
            image_array = np.rot90(image_array)  # Flip for correct orientation
        elif self.view == "Coronal":
            image_array = np.rot90(image_array)  # Flip for correct orientation
        elif self.view == "Sagittal":
            image_array = np.fliplr(np.rot90((image_array)))  # Flip for correct orientation
            
        if len(image_array.shape) == 2:
            if image_array.dtype != np.uint8:
                min_val = np.min(image_array)
                max_val = np.max(image_array)
                if min_val != max_val:  # Avoid division by zero
                    image_array = (255 * (image_array - min_val) / (max_val - min_val)).astype(np.uint8)
                else:
                    image_array = np.zeros_like(image_array, dtype=np.uint8)  # Handle flat images

            # Convert grayscale to RGB before overlaying
            image_array_rgb = np.stack([image_array] * 3, axis=-1)

            cancer_segmentation_slice = None
            organ_segmentation_slice = None

            if self.suv_overlay and self.cancer_segmentation_data is not None:
                if self.view == "Axial":
                    cancer_segmentation_slice = np.rot90(self.cancer_segmentation_data[actual_slice_index])
                elif self.view == "Coronal":
                    cancer_segmentation_slice = np.rot90(self.cancer_segmentation_data[actual_slice_index])
                elif self.view == "Sagittal":
                    cancer_segmentation_slice = np.fliplr(np.rot90((self.cancer_segmentation_data[actual_slice_index])))

            if self.organ_segmentation_overlay and self.organ_segmentation_data is not None:
                if self.view == "Axial":
                    organ_segmentation_slice = np.rot90(self.organ_segmentation_data[actual_slice_index])
                elif self.view == "Coronal":
                    organ_segmentation_slice = np.rot90(self.organ_segmentation_data[actual_slice_index])
                elif self.view == "Sagittal":
                    organ_segmentation_slice = np.fliplr(np.rot90(self.organ_segmentation_data[actual_slice_index]))

            # Overlay organ mask first, then cancer mask
            image_array_rgb = self.overlay_images(image_array_rgb, organ_segmentation=organ_segmentation_slice, cancer_segmentation=cancer_segmentation_slice)

            image = Image.fromarray(image_array_rgb, 'RGB')
        else:
            raise ValueError("Unsupported image format!")

        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

def open_image_viewers(patient_var, scan_date_var, scan_type_var, scan_region_var, cancer_segmentation_var, organ_segmentation_var, root):
    patient = patient_var.get()
    scan_date = scan_date_var.get()
    scan_type = scan_type_var.get()
    scan_region = scan_region_var.get()

    nifti_loader = NiftiLoader(base_path, patient, scan_date, scan_region)
    
    if scan_type == "CT (Normalised)":
        image_dataset = nifti_loader.load_ctres()
    elif scan_type == "PET (Normalised)":
        image_dataset = nifti_loader.load_suv()
    else:
        raise ValueError("Unknown scan type!")

    cancer_segmentation_data = nifti_loader.load_cancer_segmentation() 
    organ_segmentation_data = nifti_loader.load_organ_segmentation() 

    axial_image_data =  [image_dataset[:, :, i] for i in range(image_dataset.shape[2])]
    axial_cancer_segmentation_data = [cancer_segmentation_data[:, :, i]for i in range(cancer_segmentation_data.shape[2])] if cancer_segmentation_data is not None else None
    axial_organ_segmentation_data = [organ_segmentation_data[:, :, i] for i in range(organ_segmentation_data.shape[2])] if organ_segmentation_data is not None else None
    
    coronal_image_data = [image_dataset[:, i, :] for i in range(100, 301)]
    coronal_cancer_segmentation_data = [cancer_segmentation_data[:, i, :]  for i in range(100, 301)] if cancer_segmentation_data is not None else None
    coronal_organ_segmentation_data = [organ_segmentation_data[:, i, :] for i in range(100, 301)] if organ_segmentation_data is not None else None

    sagittal_image_data = [image_dataset[i, :, :] for i in range(100, 301)]
    sagittal_cancer_segmentation_data = [cancer_segmentation_data[i, :, :]  for i in range(100, 301)] if cancer_segmentation_data is not None else None
    sagittal_organ_segmentation_data = [organ_segmentation_data[i, :, :] for i in range(100, 301)] if organ_segmentation_data is not None else None

    # Set window positions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = int(screen_width / 3)
    window_height = int(screen_height * 0.8)  # Adjust as needed

    # Axial View on the right
    new_window_axial = tk.Toplevel(root)
    new_window_axial.geometry(f"{window_width}x{window_height}+{2 * window_width}+50")
    ImageViewer(new_window_axial, axial_image_data, "Axial " + scan_type_var.get(), cancer_segmentation_var.get(), organ_segmentation_var.get(), axial_cancer_segmentation_data, axial_organ_segmentation_data)

    # Coronal View on the left
    new_window_coronal = tk.Toplevel(root)
    new_window_coronal.geometry(f"{window_width}x{window_height}+0+50")
    ImageViewer(new_window_coronal, coronal_image_data, 
                "Coronal " + scan_type_var.get(), cancer_segmentation_var.get(), organ_segmentation_var.get(), coronal_cancer_segmentation_data, coronal_organ_segmentation_data)

    # Sagittal View in the middle
    new_window_sagittal = tk.Toplevel(root)
    new_window_sagittal.geometry(f"{window_width}x{window_height}+{window_width}+50")
    ImageViewer(new_window_sagittal, sagittal_image_data, 
                "Sagittal " + scan_type_var.get(), cancer_segmentation_var.get(), organ_segmentation_var.get(), sagittal_cancer_segmentation_data, sagittal_organ_segmentation_data)

def update_scan_dates(patient_var, scan_date_dropdown):
    selected_patient = patient_var.get()
    scan_date_dropdown['values'] = list(patients[selected_patient].keys())

def update_scan_types(patient_var, scan_date_var, scan_type_dropdown):
    scan_type_dropdown['values'] = ["CT (Normalised)", "PET (Normalised)"]
    
def update_scan_regions(patient_var, scan_date_var, scan_type_var, scan_region_dropdown):
    selected_patient = patient_var.get()
    selected_scan_date = scan_date_var.get()
    selected_scan_type = "CTRes" if scan_type_var.get() == "CT (Normalised)" else "SUV" if scan_type_var.get() == "PET (Normalised)" else scan_type_var.get()
    scan_type_path = os.path.join(base_path, selected_patient, selected_scan_date, selected_scan_type)
    
    if os.path.isfile(f"{scan_type_path}_lung_region.nii.gz"):
        scan_region_dropdown['values'] = ["All", "Lungs"]
    else:
        scan_region_dropdown['values'] = ["All"]

def setup_gui(patients):    
    root = tk.Tk()
    root.title("Patient Selection")
    root.geometry("300x450")
    dropdown_width = 100
    global patient_var
    global scan_date_var
    global scan_type_var
    global scan_region_var
    patient_var = tk.StringVar()
    scan_date_var = tk.StringVar()
    scan_type_var = tk.StringVar()
    scan_region_var = tk.StringVar()
    cancer_segmentation_var = tk.BooleanVar()
    organ_segmentation_var = tk.BooleanVar()
    
    patient_label = tk.Label(root, text="Select the patient to evaluate")
    scan_date_label = tk.Label(root, text="Select the scan date")
    scan_type_label = tk.Label(root, text="Select the scan type")
    scan_region_label = tk.Label(root, text="Select the scan region")

    patients_dropdown = ttk.Combobox(root, textvariable=patient_var, width=dropdown_width)
    patients_dropdown['values'] = list(patients.keys())
    scan_date_dropdown = ttk.Combobox(root, textvariable=scan_date_var, width=dropdown_width)
    scan_type_dropdown = ttk.Combobox(root, textvariable=scan_type_var, width=dropdown_width)
    scan_region_dropdown = ttk.Combobox(root, textvariable=scan_region_var, width=dropdown_width)

    patient_var.trace('w', lambda *args: update_scan_dates(patient_var, scan_date_dropdown))
    scan_date_var.trace('w', lambda *args: update_scan_types(patient_var, scan_date_var, scan_type_dropdown))
    scan_type_var.trace('w', lambda *args: update_scan_regions(patient_var, scan_date_var, scan_type_dropdown, scan_region_dropdown))
    
    cancer_segmentation_checkbox = tk.Checkbutton(root, text="Overlay Cancer Segmentation", variable=cancer_segmentation_var)
    organ_segmentation_checkbox = tk.Checkbutton(root, text="Overlay Organ Segmentation", variable=organ_segmentation_var)

    patient_label.pack(anchor='center', padx=10, pady=5)
    patients_dropdown.pack(anchor='center', padx=10, pady=5)
    scan_date_label.pack(anchor='center', padx=10, pady=5)
    scan_date_dropdown.pack(anchor='center', padx=10, pady=5)
    scan_type_label.pack(anchor='center', padx=10, pady=5)
    scan_type_dropdown.pack(anchor='center', padx=10, pady=5)
    scan_region_label.pack(anchor='center', padx=10, pady=5)
    scan_region_dropdown.pack(anchor='center', padx=10, pady=5)
    cancer_segmentation_checkbox.pack(anchor='center', padx=10, pady=5)
    organ_segmentation_checkbox.pack(anchor='center', padx=10, pady=5)
    
    submit_button = tk.Button(root, text="Submit", command=lambda: open_image_viewers(patient_var, scan_date_var, scan_type_var, scan_region_var, cancer_segmentation_var, organ_segmentation_var, root))
    submit_button.pack(anchor='center', padx=10, pady=20)

    root.mainloop()

def main():    
    global ORGAN_LABELS
    whole_body_ct_segmentation_metadata_path = "D:\\Datasets\\FDG-PET-CT-Lesions\\Whole Body CT Segmentation\\wholeBody_ct_segmentation\\configs\\metadata.json"
    with open(whole_body_ct_segmentation_metadata_path, 'r') as file:
        whole_body_ct_segmentation_metadata = json.load(file)
        
    # Extract the label mapping
    ORGAN_LABELS = {
        int(k): v.replace('_', ' ').title() 
        for k, v in whole_body_ct_segmentation_metadata['network_data_format']['outputs']['pred']['channel_def'].items()
    }
    
    # Add "Cancer" as the first item with label -1 (or another appropriate label if 0 is reserved)
    ORGAN_LABELS = {-1: 'Cancer', **ORGAN_LABELS}
    
    for patient_id in os.listdir(base_path):
        patient_path = os.path.join(base_path, patient_id)
        if os.path.isdir(patient_path):
            scan_dates = {}
            for scan_date in os.listdir(patient_path):
                scan_dates[scan_date] = {"CT (Normalised)": None, "PET (Normalised)": None}  # Placeholder for available scan types
            patients[patient_id] = scan_dates

    setup_gui(patients)
    
if __name__ == '__main__':
    main()