import time
import queue
import tkinter as tk
import ttkbootstrap as ttk

CONTAINER_PREP_ETA = 210  # seconds

class ProgressDialog:
    def __init__(self, master, title="Progress"):
        self.top = ttk.Toplevel(master)
        self.top.title(title)
        self.top.grab_set() # make progress dialog modal

        self.prep_label = ttk.Label(self.top, text="Preparing container: 0%")
        self.prep_label.pack(padx=10, pady=5)
        self.prep_progress = ttk.Progressbar(self.top, length=300, mode="determinate", maximum=100)
        self.prep_progress.pack(padx=10, pady=5)

        self.load_label = ttk.Label(self.top, text="Loading frames: 0%")
        self.load_label.pack(padx=10, pady=5)
        self.load_progress = ttk.Progressbar(self.top, orient="horizontal", length=300,
                                                mode="determinate", maximum=100)
        self.load_progress.pack(padx=10, pady=5)

        self.prop_label = ttk.Label(self.top, text="Propagating segmentation: 0%")
        self.prop_label.pack(padx=10, pady=5)
        self.prop_progress = ttk.Progressbar(self.top, orient="horizontal", length=300,
                                                mode="determinate", maximum=100)
        self.prop_progress.pack(padx=10, pady=5)

        # create a queue to receive progress updates
        self.progress_queue = queue.Queue()

        # container prep ETA reached => 99%. final 1% when dependencies are successfully installed
        self.prep_start_time = time.time()
        
        self.current_progress = {
            "Preparing container": 0,
            "Loading frames": 0,
            "Propagating segmentation": 0
        }

        self.max_prop = 0
    
    def update_progress(self):
        
        elapsed = time.time() - self.prep_start_time
        if self.current_progress["Preparing container"] < 100:
            computed = min(99, (elapsed / CONTAINER_PREP_ETA) * 99)
            self.current_progress["Preparing container"] = max(self.current_progress["Preparing container"], computed)
            self.prep_progress["value"] = self.current_progress["Preparing container"]
            self.prep_label.config(text=f"Preparing container: {int(self.current_progress['Preparing container'])}%")

        try:
            while True:
                stage, value = self.progress_queue.get_nowait()
                if stage == "Preparing container":
                    self.current_progress["Preparing container"] = 100
                    self.prep_progress["value"] = 100
                    self.prep_label.config(text="Preparing container: 100%")
                elif stage == "Loading frames":
                    self.current_progress["Loading frames"] = value
                    self.load_progress["value"] = value
                    self.load_label.config(text=f"Loading frames: {value}%")
                    # if loading frames has started and prepping is not at 100%, force it
                    if value > 0 and self.current_progress["Preparing container"] < 100:
                        self.current_progress["Preparing container"] = 100
                        self.prep_progress["value"] = 100
                        self.prep_label.config(text="Preparing container: 100%")
                elif stage == "Propagating segmentation":
                    self.current_progress["Propagating segmentation"] = value
                    self.prop_progress["value"] = value
                    if value < self.max_prop:
                        self.prop_label.config(text=f"Propagating segmentation backwards: {value}%")
                    else:
                        self.max_prop = value
                        self.prop_label.config(text=f"Propagating segmentation: {value}%")
                    # if propagation has started and loading frames is not at 100%, force it
                    if value > 0 and self.current_progress["Loading frames"] < 100:
                        self.current_progress["Loading frames"] = 100
                        self.load_progress["value"] = 100
                        self.load_label.config(text="Loading frames: 100%")
                elif stage == "done":
                    pass
        except queue.Empty:
            pass

        # Continue polling every 100ms until the window is destroyed.
        if self.top.winfo_exists():
            self.top.after(100, self.update_progress)
        
    def close(self):
        self.top.destroy()
