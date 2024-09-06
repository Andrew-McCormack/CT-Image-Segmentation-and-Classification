import nibabel as nib
import numpy as np
import os
from monai.bundle import ConfigParser
from skimage import measure
import pyvista as pv
from matplotlib import cm
from tkinter import Tk, Checkbutton, BooleanVar, Scrollbar, Canvas, Frame, Button
from threading import Thread, Event

# Event to synchronize the start of the legend display
render_complete = Event()

# Function to display the 3D model
def display_3d_model(legend_entries, ct_scan_data, cancer_data, visibility_dict):
    plotter = pv.Plotter()
    plotter.enable_depth_peeling()  # Enable depth peeling for better transparency rendering

    actors = {}  # Store actors corresponding to each organ for toggling visibility

    # Add cancer segmentation to the model first, render last to overlay
    if cancer_data.max() > 0:
        verts, faces, _, _ = measure.marching_cubes(cancer_data, level=0.5)
        faces_pv = np.hstack([[3] + list(face) for face in faces])
        cancer_mesh = pv.PolyData(verts, faces_pv)
        cancer_actor = plotter.add_mesh(cancer_mesh, color='#FF6EC7', opacity=1.0, label='Cancer', lighting=False)
        actors['Cancer'] = cancer_actor

    for organ, data, color in legend_entries:
        if data.max() > 0:
            verts, faces, _, _ = measure.marching_cubes(data, level=0.5)
            faces_pv = np.hstack([[3] + list(face) for face in faces])
            mesh = pv.PolyData(verts, faces_pv)
            actor = plotter.add_mesh(mesh, color=color, opacity=0.8, label=organ)
            actors[organ] = actor

    # Mask the CT scan data where organs are present, and render it last
    if ct_scan_data.max() > 0:
        verts, faces, _, _ = measure.marching_cubes(ct_scan_data, level=0.5)
        faces_pv = np.hstack([[3] + list(face) for face in faces])
        ct_mesh = pv.PolyData(verts, faces_pv)
        ct_actor = plotter.add_mesh(ct_mesh, color='gray', opacity=0.2, label='CT Scan')
        actors['CT Scan'] = ct_actor

    # Set the event once the model is rendered and the window is shown
    render_complete.set()

    plotter.show(interactive_update=True)  # Interactive mode

    # Function to update visibility based on Tkinter checkboxes
    def update_visibility():
        for organ, var in visibility_dict.items():
            actors[organ].SetVisibility(var.get())
        plotter.render()

    return update_visibility

# Function to display the scrollable legend with checkboxes in a Tkinter window
def display_legend(legend_entries, visibility_dict, update_visibility):
    # Wait for the 3D model to finish rendering before proceeding
    render_complete.wait()

    root = Tk()
    root.title("Organ Legend")

    canvas = Canvas(root)
    scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    frame = Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Checkbox for the cancer segmentation
    var = BooleanVar(value=True)
    cancer_checkbox = Checkbutton(frame, text='Cancer', bg='#FF6EC7', variable=var, command=update_visibility)
    cancer_checkbox.pack(anchor='w')
    visibility_dict['Cancer'] = var

    # Checkbox for the CT scan
    var = BooleanVar(value=True)
    ct_checkbox = Checkbutton(frame, text='CT Scan', bg='gray', variable=var, command=update_visibility)
    ct_checkbox.pack(anchor='w')
    visibility_dict['CT Scan'] = var

    for organ, _, color in legend_entries:
        var = BooleanVar(value=True)
        checkbox = Checkbutton(frame, text=organ, bg=f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}',
                               variable=var, command=update_visibility)
        checkbox.pack(anchor='w')
        visibility_dict[organ] = var

    # "Toggle All" button
    def toggle_all():
        all_checked = all(var.get() for var in visibility_dict.values())
        for var in visibility_dict.values():
            var.set(not all_checked)
        update_visibility() 

    toggle_all_button = Button(root, text="Toggle All", command=toggle_all)
    toggle_all_button.pack(side="bottom", pady=10)

    canvas.pack(side="left", fill="both", expand=True)

    root.mainloop()

# Load the NIfTI files for the scan and the segmentation maps
ct_scan_file = 'D:\\Datasets\\FDG-PET-CT-Lesions\\NIFTI\\FDG-PET-CT-Lesions\\PETCT_108c1763d4\\09-30-2004-NA-PET-CT Ganzkoerper  primaer mit KM-86848\\CTres.nii.gz'
organ_segmentation_file = 'D:\\Datasets\\FDG-PET-CT-Lesions\\NIFTI\\FDG-PET-CT-Lesions\\PETCT_108c1763d4\\09-30-2004-NA-PET-CT Ganzkoerper  primaer mit KM-86848\\mask\\CTres\\CTres_trans.nii.gz'
cancer_segmentation_file = 'D:\\Datasets\\FDG-PET-CT-Lesions\\NIFTI\\FDG-PET-CT-Lesions\\PETCT_108c1763d4\\09-30-2004-NA-PET-CT Ganzkoerper  primaer mit KM-86848\\SEG.nii.gz'

organ_segmentation = nib.load(organ_segmentation_file)
cancer_segmentation = nib.load(cancer_segmentation_file)
ct_scan = nib.load(ct_scan_file)

organ_data = organ_segmentation.get_fdata()
cancer_data = cancer_segmentation.get_fdata()
ct_scan_data = ct_scan.get_fdata()

whole_body_ct_segmentation_folder = "D:\\Datasets\\FDG-PET-CT-Lesions\\Whole Body CT Segmentation"

config_path = os.path.join(whole_body_ct_segmentation_folder, "wholeBody_ct_segmentation", "configs", "metadata.json")

config = ConfigParser()
config.read_config(config_path)

# Define bone-related organs to exclude
bone_organs = {
    'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1',
    'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8',
    'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3',
    'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5',
    'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1', 'rib_left_1',
    'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7',
    'rib_left_8', 'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12',
    'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5',
    'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10',
    'rib_right_11', 'rib_right_12', 'humerus_left', 'humerus_right', 'scapula_left',
    'scapula_right', 'clavicula_left', 'clavicula_right', 'femur_left', 'femur_right',
    'hip_left', 'hip_right', 'sacrum'
}

# Remove the "background" label and bone-related organs from organ_labels
organ_labels = {label: organ for label, organ in config["network_data_format"]["outputs"]["pred"]["channel_def"].items() if organ.lower() != "background" and organ not in bone_organs}

# Define a colormap for consistent and distinguishable colors
colormap = cm.get_cmap('tab20', len(organ_labels))

# Prepare the legend entries
legend_entries = []
visibility_dict = {}  # Dictionary to store visibility state for each organ
for i, (label, organ) in enumerate(organ_labels.items()):
    label_int = int(label)

    if np.any(organ_data == label_int):
        organ_segmented_data = np.where(organ_data == label_int, 1, 0)
        color = colormap(i)[:3]
        legend_entries.append((organ, organ_segmented_data, color))
    else:
        print(f"Label {label_int} for organ {organ} not found in segmentation data.")

# Run the PyVista 3D model
update_visibility = display_3d_model(legend_entries, ct_scan_data, cancer_data, visibility_dict)

# Run the Tkinter legend
display_legend(legend_entries, visibility_dict, update_visibility)