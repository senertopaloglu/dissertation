# exporter.py
import os
import numpy as np
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
from skimage import measure

from export_format import ExportFormat

try:
    import trimesh
except ImportError:
    trimesh = None
try:
    import nibabel as nib
except ImportError:
    nib = None

class Exporter:
    def __init__(self, main_view):
        self.main_view = main_view

    def export_3d_mesh(self) -> None:
        """
        Exports the 3D mesh of the segmentation as an STL file, from draft (in draft mode) or final segmentation result.
        
        Returns:
            None
        """
        try:
            from stl import mesh
        except ImportError:
            tk.messagebox.showerror("Export Error", "The 'numpy-stl' package is required for exporting 3D meshes.")
            return

        if self.main_view.sidebar.global_draft_mode.get():
            if not self.main_view.last_draft_result:
                tk.messagebox.showerror("Export Error", "No segmentation result available for export.")
                return
            volume_segments = self.main_view.last_draft_result
        else:
            if not self.main_view.last_result:
                tk.messagebox.showerror("Export Error", "No segmentation result available for export.")
                return
            volume_segments = self.main_view.last_result

        stl_meshes = []
        for obj_id, (verts, faces) in volume_segments.items():
            triangles = verts[faces]  # shape: (n_faces, 3, 3)
            m = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
            for i, triangle in enumerate(triangles):
                m.vectors[i] = triangle
            stl_meshes.append(m)

        if not stl_meshes:
            tk.messagebox.showerror("Export Error", "No valid mesh could be generated.")
            return

        combined_data = np.concatenate([m.data for m in stl_meshes])
        exported_mesh = mesh.Mesh(combined_data.copy())
        export_filename = tk.filedialog.asksaveasfilename(
            title="Export 3D Mesh as STL",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
            defaultextension=".stl"
        )
        if not export_filename:
            return

        exported_mesh.save(export_filename)
        tk.messagebox.showinfo("Export Successful", f"3D mesh exported as '{export_filename}'.")

    def export_view_with_mask(self) -> None:
        """
        Displays a popup for the user to select the export format and then
        exports the volume with overlayed segmentation mask as NIfTI file.

        Returns:
            None
        """
        popup = tk.Toplevel(self.main_view)
        popup.title("Choose Export Color Format")
        popup.transient(self.main_view)

        instruction_label = tk.Label(popup, text="Select export format for the exported view:")
        instruction_label.pack(pady=10)

        export_var = tk.StringVar(value=ExportFormat.BINARY.value)
        for fmt in ExportFormat:
            rb = tk.Radiobutton(popup, text=str(fmt), variable=export_var, value=fmt.value)
            rb.pack(anchor="w", padx=20)

        def on_confirm():
            chosen_format = ExportFormat(export_var.get())
            popup.destroy()
            self._export_view_with_mask_process(chosen_format)

        confirm_button = tk.Button(popup, text="OK", command=on_confirm)
        confirm_button.pack(pady=10)

    def _export_view_with_mask_process(self, chosen_format: ExportFormat) -> None:
        """
        Helper function: handles overlay and transformation of segmentation mask onto slices of original 3d volume.

        Args:
            chosen_format (ExportFormat): The chosen export format (BINARY, GRAYSCALE, or color).

        Returns:
            None
        """
        alpha = 0.4
        original_np = np.asarray(self.main_view.model.image)
        active_index = self.main_view.sidebar.tabControl.index("current")
        current_tab = self.main_view.sidebar.tabs[active_index]
        num_slices = original_np.shape[active_index]

        if self.main_view.sidebar.global_draft_mode.get():
            mask_dict = self.main_view._get_mask(active_index, is_draft=True)
        else:
            mask_dict = self.main_view._get_mask(active_index)

        if mask_dict is None:
            tk.messagebox.showerror("Export Error", "No segmentation mask available for selected view.")
            return

        composite_slices = []
        grayscale_mapping = {1: 63, 2: 252, 3: 189, 4: 126, 5: 111, 6: 96, 7: 71, 8: 56, 9: 41, 10: 26}

        for i in range(num_slices):
            if active_index == 0:
                orig_slice = original_np[i, :, :]
            elif active_index == 1:
                orig_slice = original_np[:, i, :]
            elif active_index == 2:
                orig_slice = original_np[:, :, i]
                orig_slice = np.rot90(orig_slice, k=-1)
                orig_slice = np.fliplr(orig_slice)
            
            slice_min, slice_max = orig_slice.min(), orig_slice.max()
            if slice_max > slice_min:
                norm_slice = (orig_slice - slice_min) / (slice_max - slice_min)
            else:
                norm_slice = orig_slice * 0
            norm_slice_255 = (norm_slice * 255).astype(np.uint8)

            rgb = np.stack([norm_slice_255] * 3, axis=-1).astype(np.float32)
            if chosen_format in (ExportFormat.BINARY, ExportFormat.GRAYSCALE):
                composite = np.zeros_like(rgb)
            else:
                composite = rgb.copy()

            if i in mask_dict:
                for obj_id, mask in sorted(mask_dict[i].items(), key=lambda item: item[0]):
                    if active_index == 2:
                        mask = np.squeeze(mask.astype(np.float32))
                        mask = np.rot90(mask, k=-1)
                        mask = np.fliplr(mask)
                        mask_expanded = mask[..., None]
                    else:
                        mask_expanded = np.expand_dims(mask.astype(np.float32), axis=-1)
                    if chosen_format == ExportFormat.BINARY:
                        binary_mask = (mask_expanded > 0).squeeze()
                        composite[binary_mask] = 255
                    elif chosen_format == ExportFormat.GRAYSCALE:
                        gray_val = grayscale_mapping.get(obj_id, 0)
                        overlay = np.full_like(composite, gray_val)
                        composite = np.where(mask_expanded > 0, (1 - alpha) * composite + alpha * overlay, composite)
                    else:
                        colour_name = self.main_view.pointer_color_mapping.get(obj_id, "red")
                        colour_rgb = np.array(mcolors.to_rgb(colour_name)) * 255
                        overlay = np.full_like(composite, colour_rgb)
                        composite = np.where(mask_expanded > 0, (1 - alpha) * composite + alpha * overlay, composite)
                    if chosen_format != ExportFormat.BINARY:
                        composite = composite.clip(0, 255)
            
            composite_slices.append(composite.astype(np.uint8))
        
        composite_volume = np.stack(composite_slices, axis=0)
        if composite_volume.ndim == 5 and composite_volume.shape[1] == 1:
            composite_volume = composite_volume.squeeze(axis=1)
        
        if active_index == 1:
            composite_volume = composite_volume.transpose(1, 0, 2, 3)
        if active_index == 2:
            composite_volume = composite_volume.transpose(2, 1, 0, 3)

        if nib is None:
            tk.messagebox.showerror("Export Error", "The 'nibabel' package is required for exporting 3D NIfTI files.")
            return

        export_filename = tk.filedialog.asksaveasfilename(
            title="Export 3D Image with Segmentation Overlay as NIfTI",
            defaultextension=".nii",
            filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")]
        )
        if not export_filename:
            return

        nii_img = nib.Nifti1Image(composite_volume, affine=np.eye(4))
        nib.save(nii_img, export_filename)
        tk.messagebox.showinfo("Export Successful", f"Image with segmentation overlay exported as:\n{export_filename}")