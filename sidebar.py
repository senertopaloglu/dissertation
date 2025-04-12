import os
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap import Frame, Label, Button, Notebook, OptionMenu, Checkbutton
import matplotlib.pyplot as plt


class Sidebar(Frame):
    def __init__(self, parent: tk.Widget, model, view, controller):
        super().__init__(parent, padding=10)
        self.model = model
        self.view = view
        self.controller = controller
        
        self.undo_icon = tk.PhotoImage(file=os.path.join("images", "undo.png"))
        self.redo_icon = tk.PhotoImage(file=os.path.join("images", "redo.png"))
        self.segment_icon = tk.PhotoImage(file=os.path.join("images", "segment.png"))
        self.import_icon = tk.PhotoImage(file=os.path.join("images", "import.png"))
        self.export_icon = tk.PhotoImage(file=os.path.join("images", "export.png"))
        self.merge_icon = tk.PhotoImage(file=os.path.join("images", "merge.png"))

        self.global_draft_mode = ttk.BooleanVar(value=False)
        self.global_segmentation_var = ttk.BooleanVar(value=False)
        
        self._build_sidebar()

    def _build_sidebar(self) -> None:
        """
        Build and initialize the sidebar UI components.

        This method creates and configures the import/export buttons, tab controls,
        and various UI elements (like the pointer color option menu, points listbox,
        and undo/redo buttons) for each tab. It also sets up styling and event bindings
        for menu updates and state changes.
        """
        # Import and Export Buttons
        btn_import_nifti = Button(self, text="Import NIfTI File", command=self._import_nifti,
                             bootstyle="primary", image=self.import_icon, compound="left")
        btn_import_nifti.pack(fill="x", pady=(5,2))

        btn_import_dicom = Button(self, text="Import DICOM Folder", command=self._import_dicom,
                                bootstyle="primary", image=self.import_icon, compound="left")
        btn_import_dicom.pack(fill="x", pady=(2,5))

        export_style = ttk.Style()
        export_style.configure("DarkGrey.TButton",
                               background="#555555",
                               foreground="white",
                               bordercolor="#555555",
                               borderwidth=0,
                               focusthickness=0,
                               relief="flat",
                               padding=(10, 8),
                               anchor="center",
                               justify="center")
        export_style.map("DarkGrey.TButton", background=[("active", "#666666")])
        btn_export = Button(self, text="Export 3D Mesh Model", command=self._export_3d_mesh,
                            image=self.export_icon, compound="left", style="DarkGrey.TButton")
        btn_export.pack(fill="x", pady=5)

        # Tab control and style
        style = ttk.Style()
        style.configure("DarkerTabs.TNotebook", background="white", tabmargins=[2, 5, 2, 0])
        style.configure("DarkerTabs.TNotebook.Tab",
                        background="#DDDDDD",
                        padding=[10, 4],
                        font=("TkDefaultFont", 10))
        style.map("DarkerTabs.TNotebook.Tab",
                  background=[("selected", "white")],
                  foreground=[("selected", "black")])
        style.configure("DarkGreen.TButton",
                        background="#006400",
                        foreground="white",
                        relief="flat",
                        borderwidth=0,
                        padding=(10, 8),
                        anchor="center",
                        justify="center")
        style.map("DarkGreen.TButton", background=[("active", "#228B22")])

        tabControl = Notebook(self, style="DarkerTabs.TNotebook")
        tab1 = Frame(tabControl)
        tab2 = Frame(tabControl)
        tab3 = Frame(tabControl)
        tabControl.add(tab1, text="Axial")
        tabControl.add(tab2, text="Coronal")
        tabControl.add(tab3, text="Sagittal")
        tabControl.pack(fill="x", pady=(0,5))
        self.tabControl = tabControl
        self.tabs = [tab1, tab2, tab3]

        # Initialize each tab’s properties and build its UI.
        for i, tab in enumerate(self.tabs):
            tab.style_name = f"PointerColor.TMenubutton.Tab{i}"
            style.layout(tab.style_name, style.layout("TMenubutton"))
            tab.pointer_color_var = None
            tab.pointer_color_optionmenu = None
            tab.points_listbox = None
            tab.draft_points = []
            tab.points = []
            tab.draft_line_objects = []
            tab.line_objects = []
            tab.draft_undo_stack =[]
            tab.undo_stack = []
            tab.draft_redo_stack = []
            tab.redo_stack = []

        for tab in self.tabs:
            content_frame = Frame(tab, padding=(10, 5))
            content_frame.pack(fill="both", expand=True)

            pointer_color_var = ttk.StringVar(value="Red")
            tab.pointer_color_var = pointer_color_var

            pos_click_var = ttk.BooleanVar(value=True)
            tab.pos_click_var = pos_click_var
            pos_click_checkbox = Checkbutton(content_frame, text="Positive Click", variable=pos_click_var)
            if not self.model.image:
                pos_click_checkbox.config(state="disabled")
            pos_click_checkbox.pack(pady=(2, 0))
            tab.pos_click_checkbox = pos_click_checkbox



            pointer_label = Label(content_frame, text="Pointer colour:")
            pointer_label.pack(pady=(2, 2))

            colors = ["Red", "Blue", "Green", "Orange", "Purple", "Cyan", "Magenta", "Teal", "Black", "Gray"]

            tab.pointer_color_optionmenu = OptionMenu(content_frame, pointer_color_var, "")
            tab.pointer_color_optionmenu.pack(fill="x", pady=(0,0))
            tab.pointer_color_optionmenu.configure(textvariable=pointer_color_var)
            menu = tab.pointer_color_optionmenu["menu"]
            menu.delete(0, "end")
            for color in colors:
                menu.add_command(label=color,
                                 foreground=color.lower(),
                                 background="white",
                                 activeforeground="white",
                                 activebackground=color.lower(),
                                 command=lambda c=color, var=pointer_color_var: var.set(c))
            pointer_color_var.set("Red")

            def update_option_menu_color(*args, current_tab=tab):
                selected = current_tab.pointer_color_var.get().lower()
                style.configure(current_tab.style_name,
                                foreground=selected,
                                background="white",
                                relief="solid",
                                borderwidth=1)
                style.map(current_tab.style_name,
                          background=[("active", "white"), ("pressed", "white")])
                current_tab.pointer_color_optionmenu.configure(style=current_tab.style_name)
                points_for_color = [pt for pt in current_tab.points if pt[2].lower() == selected]
                if points_for_color:
                    current_tab.pos_click_checkbox.config(state="normal")
                else:
                    current_tab.pos_click_var.set(True)
                    current_tab.pos_click_checkbox.config(state="disabled")
            pointer_color_var.trace_add("write", update_option_menu_color)
            update_option_menu_color()

            separator1 = ttk.Separator(content_frame, orient="horizontal")
            separator1.pack(fill="x", pady=10)

            tab.draft_selection_var = self.global_draft_mode
            draft_check = Checkbutton(
                content_frame,
                text="Draft Mode",
                variable=tab.draft_selection_var,
                command=lambda tab=tab, axis=self.tabs.index(tab): (
                    self.view.update_all_views(),
                    self.view.update_mesh_view("AXIAL" if axis == 0 else "CORONAL" if axis == 1 else "SAGITTAL")
                )
            )
            draft_check.pack(pady=(0, 2))

            draft_points_label = Label(content_frame, text="Draft Points")
            draft_points_label.pack(pady=(0, 2))

            draft_points_frame = Frame(content_frame)
            draft_points_frame.pack(fill="x")
            draft_scrollbar = ttk.Scrollbar(draft_points_frame, orient="vertical")
            tab.draft_points_listbox = tk.Listbox(draft_points_frame, height=5, yscrollcommand=draft_scrollbar.set)
            tab.draft_points_listbox.pack(side="left", fill="x", expand=True)
            draft_scrollbar.config(command=tab.draft_points_listbox.yview)
            draft_scrollbar.pack(side="right", fill="y")

            draft_undo_redo_frame = Frame(content_frame)
            draft_undo_redo_frame.pack(fill="x", pady=2)
            draft_undo_redo_frame.columnconfigure(0, weight=1)
            draft_undo_redo_frame.columnconfigure(1, weight=1)

            tab.draft_btn_undo = Button(draft_undo_redo_frame, text="Undo", command=lambda: self._on_undo_click(True),
                                        bootstyle="info", image=self.undo_icon, compound="left")
            tab.draft_btn_undo.grid(row=0, column=0, sticky="ew", padx=(0, 5))
            tab.draft_btn_redo = Button(draft_undo_redo_frame, text="Redo", command=lambda: self._on_redo_click(True),
                                        bootstyle="info", image=self.redo_icon, compound="left")
            tab.draft_btn_redo.grid(row=0, column=1, sticky="ew")

            tab.merge_drafts_btn = Button(content_frame, text="Merge Drafts into Selected",
                                          command=lambda: self.controller.merge_drafts() if self.controller else None,
                                          image=self.merge_icon, compound="left",
                                          style="DarkGreen.TButton")
            tab.merge_drafts_btn.pack(fill="x", pady=(2, 2))

            points_label = Label(content_frame, text="Selected Points")
            points_label.pack(pady=(0, 2))

            points_frame = Frame(content_frame)
            points_frame.pack(fill="x")

            scrollbar = ttk.Scrollbar(points_frame, orient="vertical")
            tab.points_listbox = tk.Listbox(points_frame, height=5, yscrollcommand=scrollbar.set)
            tab.points_listbox.pack(side="left", fill="x", expand=True)
            scrollbar.config(command=tab.points_listbox.yview)
            scrollbar.pack(side="right", fill="y")

            undo_redo_frame = Frame(content_frame)
            undo_redo_frame.pack(fill="x", pady=(2, 0))
            undo_redo_frame.columnconfigure(0, weight=1)
            undo_redo_frame.columnconfigure(1, weight=1)

            btn_undo = Button(undo_redo_frame, text="Undo", command=lambda: self._on_undo_click(False),
                              bootstyle="info", image=self.undo_icon, compound="left")
            btn_undo.grid(row=0, column=0, sticky="ew", padx=(0, 5))
            btn_redo = Button(undo_redo_frame, text="Redo", command=lambda: self._on_redo_click(False),
                              bootstyle="info", image=self.redo_icon, compound="left")
            btn_redo.grid(row=0, column=1, sticky="ew")

            separator2 = ttk.Separator(content_frame, orient="horizontal")
            separator2.pack(fill="x", pady=10)

            tab.global_segmentation_checkbox = Checkbutton(
                content_frame,
                text="Global view segmentation\n(show on axial view)",
                variable=self.global_segmentation_var,
                state="disabled"  # disabled by default
            )
            tab.global_segmentation_checkbox.pack(fill="x", pady=(0, 2))

            tab.btn_segment = Button(content_frame, text="Segment Image",
                                     command=self._segment_image,
                                     image=self.segment_icon, compound="left",
                                     style="DarkGreen.TButton")
            tab.btn_segment.pack(fill="x", pady=2)

            tab.btn_auto_seg = ttk.Button(content_frame,
                                          text="Apply Multiresolution\nSegmentation",
                                          command=lambda t=tab: self.controller.multiresolution_segmentation(t) if self.model.image else None,
                                          image=self.segment_icon, compound="left", bootstyle="success")
            tab.btn_auto_seg.pack(fill="x", pady=2)

            tab.btn_export_view = ttk.Button(content_frame, text="Export View with\nSegmentation Masks",
                                             command=lambda: self._export_view_with_mask(),
                                             image=self.export_icon, compound="left", style="DarkGrey.TButton")
            tab.btn_export_view.pack(fill="x", pady=2)

    # The functions below forward user actions to the controller.
    def _import_nifti(self) -> None:
        if self.view:
            self.view.import_nifti()
    
    def _import_dicom(self) -> None:
        if self.view:
            self.view.import_dicom()
            
    def _import_image(self) -> None:
        if self.view:
            self.view._import_image()

    def _export_3d_mesh(self) -> None:
        if self.view:
            self.view._export_3d_mesh()

    def _segment_image(self) -> None:
        if self.view:
            self.view._segment_image()

    def _export_view_with_mask(self) -> None:
        if self.view:
            self.view._export_view_with_mask()

    def _on_undo_click(self, is_draft: bool = False) -> None:
        if self.view:
            self.view._on_undo_click(is_draft)

    def _on_redo_click(self, is_draft: bool = False) -> None:
        if self.view:
            self.view._on_redo_click(is_draft)
