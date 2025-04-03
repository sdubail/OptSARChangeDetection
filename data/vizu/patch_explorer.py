#!/usr/bin/env python
"""
Efficient SAR-Optical patch explorer using Tkinter.
Allows browsing through positive and negative patch pairs with robust navigation.
"""

import argparse
import json
import os
from pathlib import Path
from threading import Thread
from tkinter import (
    BOTH,
    BOTTOM,
    HORIZONTAL,
    LEFT,
    RIGHT,
    TOP,
    Button,
    Entry,
    Frame,
    Label,
    OptionMenu,
    Scale,
    StringVar,
    Tk,
    X,
    Y,
)

import h5py
import numpy as np
from PIL import Image, ImageTk


class AsyncDataLoader:
    """Handles asynchronous loading of patches from HDF5 file."""

    def __init__(self, h5_path, metadata):
        self.h5_path = h5_path
        self.metadata = metadata
        self.h5_file = None
        self.open_h5()

    def open_h5(self):
        """Open the HDF5 file if not already open."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

    def close_h5(self):
        """Close the HDF5 file if open."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def load_patch(self, index, callback):
        """Load a patch asynchronously and call the callback when done."""

        def _load():
            try:
                self.open_h5()
                meta = self.metadata[index]
                h5_idx = meta["index"]

                pre_patch = self.h5_file["pre_patches"][h5_idx]
                post_patch = self.h5_file["post_patches"][h5_idx]
                label = self.h5_file["labels"][h5_idx]

                # Ensure label is 2D
                if label.ndim == 3 and label.shape[2] == 1:
                    label = label.squeeze()

                # Normalize images for display
                pre_patch_norm = self.normalize_for_display(pre_patch)
                print("SHAPE:", post_patch.shape)
                post_patch_norm = self.normalize_for_display(post_patch)

                # Convert to 8-bit for PIL
                pre_patch_norm = (pre_patch_norm * 255).astype(np.uint8)
                post_patch_norm = (post_patch_norm * 255).astype(np.uint8)

                # Create visualization for damage labels
                label_viz = self.visualize_label(label)

                # Return data through callback
                callback(pre_patch_norm, post_patch_norm, label_viz, meta)
            except Exception as e:
                print(f"Error loading patch: {e}")
                callback(None, None, None, meta)

        # Start loading in a separate thread
        Thread(target=_load).start()

    def normalize_for_display(self, img):
        """Normalize an image for display."""
        img_min = np.percentile(
            img, 1
        )  # 1st percentile instead of min to handle outliers
        img_max = np.percentile(img, 99)  # 99th percentile instead of max

        if img_max > img_min:
            img_norm = np.clip((img - img_min) / (img_max - img_min), 0, 1)
            return img_norm
        return img.astype(float) / 255.0

    def visualize_label(self, label):
        """Create a color visualization of a label."""
        # Define colors for damage levels (0 to 4)
        colors = np.array(
            [
                [0, 0, 0],  # 0: No damage (black)
                [0, 255, 0],  # 1: Minor damage (green)
                [255, 255, 0],  # 2: Moderate damage (yellow)
                [255, 127, 0],  # 3: Major damage (orange)
                [255, 0, 0],  # 4: Destroyed (red)
            ]
        )

        # Create RGB image
        h, w = label.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Fill with colors based on label values
        for i in range(5):  # 5 damage levels
            mask = label == i
            if np.any(mask):
                rgb[mask] = colors[i]

        return rgb

    def __del__(self):
        """Close the HDF5 file when the object is destroyed."""
        self.close_h5()


class PatchExplorerApp:
    """Main application for exploring SAR-Optical patch pairs."""

    def __init__(self, root, patch_dir, split="train"):
        self.root = root
        self.patch_dir = Path(patch_dir)
        self.split = split

        # Set window title and size
        root.title(f"SAR-Optical Patch Explorer - {split}")
        root.geometry("1200x800")

        print(f"Loading {split} dataset from {patch_dir}...")

        # Load metadata
        with open(self.patch_dir / f"{split}_metadata.json", "r") as f:
            self.metadata = json.load(f)

        # Get positive and negative indices
        self.pos_indices = [
            i for i, m in enumerate(self.metadata) if m.get("is_positive")
        ]
        self.neg_indices = [
            i for i, m in enumerate(self.metadata) if not m.get("is_positive")
        ]

        print(
            f"Found {len(self.pos_indices)} positive and {len(self.neg_indices)} negative pairs"
        )

        # Initialize data loader
        h5_path = self.patch_dir / f"{split}_patches.h5"
        self.data_loader = AsyncDataLoader(h5_path, self.metadata)

        # Initialize state
        self.current_mode = "positive"  # or "negative"
        self.curr_idx = 0
        self.indices = self.pos_indices

        # Create UI components
        self.create_widgets()

        # Load first patch
        self.update_display()

    def create_widgets(self):
        """Create the UI widgets."""
        # Main frame
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Top info panel
        info_frame = Frame(main_frame)
        info_frame.pack(fill=X, side=TOP, pady=(0, 10))

        # Status label
        self.status_var = StringVar()
        self.status_var.set("Loading...")
        status_label = Label(
            info_frame, textvariable=self.status_var, font=("Helvetica", 12, "bold")
        )
        status_label.pack(side=LEFT)

        # Pair type selector
        mode_frame = Frame(info_frame)
        mode_frame.pack(side=RIGHT)

        mode_label = Label(mode_frame, text="View:")
        mode_label.pack(side=LEFT, padx=(0, 5))

        self.mode_var = StringVar(value="Positive Pairs (No Damage)")
        mode_options = ["Positive Pairs (No Damage)", "Negative Pairs (With Damage)"]
        mode_menu = OptionMenu(
            mode_frame, self.mode_var, *mode_options, command=self.on_mode_change
        )
        mode_menu.pack(side=LEFT)

        # Image display area
        image_frame = Frame(main_frame)
        image_frame.pack(fill=BOTH, expand=True)

        # Image labels
        self.pre_label = Label(image_frame, text="Pre-event (Optical)")
        self.pre_label.grid(row=0, column=0, padx=5, pady=5)

        self.post_label = Label(image_frame, text="Post-event (SAR)")
        self.post_label.grid(row=0, column=1, padx=5, pady=5)

        self.label_label = Label(image_frame, text="Damage Label")
        self.label_label.grid(row=0, column=2, padx=5, pady=5)

        # Image panels
        self.pre_panel = Label(image_frame)
        self.pre_panel.grid(row=1, column=0, padx=5, pady=5)

        self.post_panel = Label(image_frame)
        self.post_panel.grid(row=1, column=1, padx=5, pady=5)

        self.label_panel = Label(image_frame)
        self.label_panel.grid(row=1, column=2, padx=5, pady=5)

        # Metadata display
        self.meta_var = StringVar()
        meta_label = Label(
            image_frame,
            textvariable=self.meta_var,
            justify=LEFT,
            font=("Helvetica", 10),
        )
        meta_label.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky="w")

        # Navigation frame
        nav_frame = Frame(main_frame)
        nav_frame.pack(fill=X, side=BOTTOM, pady=(10, 0))

        # Navigation buttons
        btn_prev_100 = Button(
            nav_frame, text="<<< -100", command=lambda: self.move_by(-100)
        )
        btn_prev_100.pack(side=LEFT, padx=5)

        btn_prev_10 = Button(
            nav_frame, text="<< -10", command=lambda: self.move_by(-10)
        )
        btn_prev_10.pack(side=LEFT, padx=5)

        btn_prev = Button(nav_frame, text="< Prev", command=lambda: self.move_by(-1))
        btn_prev.pack(side=LEFT, padx=5)

        btn_next = Button(nav_frame, text="Next >", command=lambda: self.move_by(1))
        btn_next.pack(side=LEFT, padx=5)

        btn_next_10 = Button(nav_frame, text="+10 >>", command=lambda: self.move_by(10))
        btn_next_10.pack(side=LEFT, padx=5)

        btn_next_100 = Button(
            nav_frame, text="+100 >>>", command=lambda: self.move_by(100)
        )
        btn_next_100.pack(side=LEFT, padx=5)

        # Jump to index
        jump_frame = Frame(nav_frame)
        jump_frame.pack(side=RIGHT)

        jump_label = Label(jump_frame, text="Jump to index:")
        jump_label.pack(side=LEFT, padx=(0, 5))

        self.jump_entry = Entry(jump_frame, width=10)
        self.jump_entry.pack(side=LEFT, padx=(0, 5))
        self.jump_entry.bind("<Return>", self.on_jump)

        jump_button = Button(jump_frame, text="Go", command=self.on_jump)
        jump_button.pack(side=LEFT)

        # Slider for navigation
        slider_frame = Frame(main_frame)
        slider_frame.pack(fill=X, side=BOTTOM, pady=(5, 0))

        self.slider = Scale(
            slider_frame,
            from_=0,
            to=max(len(self.pos_indices), len(self.neg_indices)) - 1,
            orient=HORIZONTAL,
            length=600,
            command=self.on_slider_change,
        )
        self.slider.pack(fill=X, expand=True)

        # Set slider to 0
        self.slider.set(0)

    def on_mode_change(self, *args):
        """Handle mode change (positive/negative pairs)."""
        mode_text = self.mode_var.get()
        if "Positive" in mode_text and self.current_mode != "positive":
            self.current_mode = "positive"
            self.indices = self.pos_indices
            self.curr_idx = (
                min(self.curr_idx, len(self.indices) - 1) if self.indices else 0
            )
            self.slider.config(to=len(self.indices) - 1 if self.indices else 0)
            self.update_display()
        elif "Negative" in mode_text and self.current_mode != "negative":
            self.current_mode = "negative"
            self.indices = self.neg_indices
            self.curr_idx = (
                min(self.curr_idx, len(self.indices) - 1) if self.indices else 0
            )
            self.slider.config(to=len(self.indices) - 1 if self.indices else 0)
            self.update_display()

    def on_slider_change(self, value):
        """Handle slider change."""
        try:
            idx = int(float(value))
            if idx != self.curr_idx and 0 <= idx < len(self.indices):
                self.curr_idx = idx
                self.update_display()
        except Exception as e:
            print(f"Error in slider change: {e}")

    def on_jump(self, event=None):
        """Handle jump to index."""
        try:
            idx = int(self.jump_entry.get())
            if 0 <= idx < len(self.indices):
                self.curr_idx = idx
                self.slider.set(idx)
                self.update_display()
            else:
                print(f"Index out of range: 0 <= {idx} < {len(self.indices)}")
        except ValueError:
            print("Please enter a valid number")

    def move_by(self, amount):
        """Move by a specific amount of indices."""
        new_idx = self.curr_idx + amount
        new_idx = max(0, min(new_idx, len(self.indices) - 1))
        if new_idx != self.curr_idx:
            self.curr_idx = new_idx
            self.slider.set(new_idx)
            self.update_display()

    def update_display(self):
        """Update the display with the current patch."""
        if not self.indices:
            self.status_var.set(
                f"No {'positive' if self.current_mode == 'positive' else 'negative'} pairs found"
            )
            return

        # Update status
        self.status_var.set(
            f"Loading patch {self.curr_idx + 1} of {len(self.indices)}..."
        )

        # Get the actual index from our list of indices
        global_idx = self.indices[self.curr_idx]

        # Load data asynchronously
        self.data_loader.load_patch(global_idx, self.update_images)

    def update_images(self, pre_patch, post_patch, label_viz, meta):
        """Update the UI with loaded images."""
        if pre_patch is None or post_patch is None or label_viz is None:
            self.status_var.set("Error loading patch")
            return

        # Convert numpy arrays to PhotoImage
        pre_img = ImageTk.PhotoImage(Image.fromarray(pre_patch))
        post_img = ImageTk.PhotoImage(Image.fromarray(post_patch))
        label_img = ImageTk.PhotoImage(Image.fromarray(label_viz))

        # Update image panels
        self.pre_panel.configure(image=pre_img)
        self.pre_panel.image = pre_img  # Keep a reference

        self.post_panel.configure(image=post_img)
        self.post_panel.image = post_img

        self.label_panel.configure(image=label_img)
        self.label_panel.image = label_img

        # Update metadata text
        pair_type = (
            "POSITIVE PAIR (No Damage)"
            if meta["is_positive"]
            else f"NEGATIVE PAIR (Damage Ratio: {meta['damage_ratio']:.2f})"
        )

        meta_text = (
            f"{pair_type}\n"
            f"Image ID: {meta['image_id']}\n"
            f"Position: {meta['position']}"
        )
        self.meta_var.set(meta_text)

        # Update status
        self.status_var.set(
            f"{'Positive' if self.current_mode == 'positive' else 'Negative'} pair "
            f"{self.curr_idx + 1} of {len(self.indices)}"
        )

    def on_closing(self):
        """Handle window closing."""
        if hasattr(self, "data_loader"):
            self.data_loader.close_h5()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive explorer for SAR-Optical patch pairs"
    )
    parser.add_argument(
        "--patch_dir",
        type=str,
        default="data/processed_patches",
        help="Directory containing preprocessed patches",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to use",
    )
    args = parser.parse_args()

    # Create the Tkinter application
    root = Tk()
    app = PatchExplorerApp(root, args.patch_dir, args.split)

    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()
