#!/usr/bin/env python
"""
Full image explorer for SAR-Optical change detection dataset.
Allows browsing and analysis of full satellite images with zoom and pan capabilities.
Added keyboard navigation: 'n' for next, 'w' to mark as 'w', 's' to mark as 's'.
"""

import argparse
import os
import threading
from pathlib import Path
from tkinter import (
    BOTH,
    BOTTOM,
    HORIZONTAL,
    LEFT,
    RIGHT,
    TOP,
    Button,
    Canvas,
    Entry,
    Frame,
    Label,
    OptionMenu,
    Scale,
    Scrollbar,
    StringVar,
    Tk,
    X,
    Y,
    ttk,
    simpledialog,
    messagebox,
)

import numpy as np
import rasterio
from PIL import Image, ImageTk

# Set this to handle larger images
Image.MAX_IMAGE_PIXELS = None


class SatelliteImageLoader:
    """Handles loading and processing of satellite images."""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.cache = {}  # Cache for loaded images
        self.max_cache_size = 5  # Max number of images to keep in cache

    def load_image_pair(self, image_id, split, callback=None):
        """Load a pre-event, post-event, target image triplet."""
        try:
            # Define paths
            pre_path = (
                self.root_dir / split / "pre-event" / f"{image_id}_pre_disaster.tif"
            )
            post_path = (
                self.root_dir / split / "post-event" / f"{image_id}_post_disaster.tif"
            )
            target_path = (
                self.root_dir / split / "target" / f"{image_id}_building_damage.tif"
            )

            # Start a thread to load the images
            threading.Thread(
                target=self._load_images_thread,
                args=(pre_path, post_path, target_path, image_id, callback),
            ).start()

            return True
        except Exception as e:
            print(f"Error loading image pair: {e}")
            if callback:
                callback(None, None, None, None)
            return False

    def _load_images_thread(self, pre_path, post_path, target_path, image_id, callback):
        """Thread function to load images without blocking the UI."""
        try:
            # Load pre-event optical image
            pre_img = self._load_tiff(pre_path)

            # Load post-event SAR image
            post_img = self._load_tiff(post_path)

            # Load target/damage image
            target_img = self._load_tiff(target_path)

            # Get image info
            with rasterio.open(pre_path) as src:
                info = {
                    "width": src.width,
                    "height": src.height,
                    "crs": src.crs.to_string() if src.crs else "None",
                    "bands": src.count,
                    "dtype": src.dtypes[0],
                    "nodata": src.nodata,
                }

            # Cache the images
            self._update_cache(
                image_id,
                {"pre": pre_img, "post": post_img, "target": target_img, "info": info},
            )

            # Call the callback with the loaded images
            if callback:
                callback(pre_img, post_img, target_img, info)

        except Exception as e:
            print(f"Error in load thread: {e}")
            if callback:
                callback(None, None, None, None)

    def _load_tiff(self, path):
        """Load a TIFF image using rasterio."""
        try:
            with rasterio.open(path) as src:
                if src.count == 1:
                    # Single band (grayscale)
                    img = src.read(1)
                    # Handle NaN values
                    if np.issubdtype(img.dtype, np.floating):
                        img = np.nan_to_num(img)
                    return img
                else:
                    # Multi-band (RGB)
                    img = np.zeros(
                        (src.height, src.width, min(3, src.count)), dtype=np.float32
                    )
                    for i in range(min(3, src.count)):
                        band = src.read(i + 1)
                        if np.issubdtype(band.dtype, np.floating):
                            band = np.nan_to_num(band)
                        img[:, :, i] = band
                    return img
        except Exception as e:
            print(f"Error loading TIFF {path}: {e}")
            raise

    def _update_cache(self, image_id, data):
        """Update the image cache."""
        # Add to cache
        self.cache[image_id] = data

        # If cache is too large, remove oldest items
        if len(self.cache) > self.max_cache_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]

    def get_cached(self, image_id):
        """Get an image from the cache if available."""
        return self.cache.get(image_id, None)


class ImageCanvas(Canvas):
    """Enhanced canvas for displaying and interacting with satellite images."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Image display properties
        self.image = None  # Original PIL image
        self.photo_image = None  # PhotoImage for display
        self.image_id = None  # Canvas ID of the displayed image

        # Zoom and pan state
        self.zoom_level = 1.0
        self.min_zoom = 0.05  # Allow more zoom out
        self.max_zoom = 10.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.start_x = 0
        self.start_y = 0

        # Image data
        self.numpy_image = None  # Original numpy image
        self.display_range = (0, 255)  # Range for display scaling
        
        # Bind to configure event to handle canvas resizing
        self.bind("<Configure>", self.on_resize)

        # Setup interactions
        self.bind("<ButtonPress-1>", self.on_button_press)
        self.bind("<B1-Motion>", self.on_button_motion)
        self.bind("<ButtonRelease-1>", self.on_button_release)
        self.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down
        
    def on_resize(self, event):
        """Handle canvas resize event."""
        # Only refit if we already have an image
        if self.image is not None:
            # Wait a bit to ensure the resize is complete
            self.after(100, self.fit_to_canvas)

    def set_image(self, np_image, title=None):
        """Set a new image to display."""
        if np_image is None:
            self.delete("all")
            self.create_text(
                self.winfo_width() // 2,
                self.winfo_height() // 2,
                text="No image available",
                fill="white",
            )
            return

        # Store the numpy image
        self.numpy_image = np_image

        # Normalize the image to 0-255 range for display
        self.update_display()

        # Calculate optimal zoom level to fit the image in the canvas
        self.fit_to_canvas()

        # Add title if provided
        if title:
            self.delete("title")
            self.create_text(
                10,
                10,
                text=title,
                anchor="nw",
                fill="white",
                tags=("title"),
                font=("Helvetica", 12, "bold"),
            )

    def update_display(self, display_min=None, display_max=None):
        """Update the displayed image with new range."""
        if self.numpy_image is None:
            return

        # Update display range if provided
        if display_min is not None and display_max is not None:
            self.display_range = (display_min, display_max)

        # Apply display range
        img_norm = self.normalize_for_display(
            self.numpy_image, self.display_range[0], self.display_range[1]
        )

        # Convert to PIL Image
        if len(img_norm.shape) == 2:
            # Grayscale
            self.image = Image.fromarray(img_norm)
        else:
            # RGB
            self.image = Image.fromarray(img_norm)

        # Display the image
        self.zoom_image()

    def normalize_for_display(self, img, display_min, display_max):
        """Normalize image for display with the given range."""
        # Get image properties
        if len(img.shape) == 2:
            # Grayscale
            img_min = np.percentile(img, 2)
            img_max = np.percentile(img, 98)
        else:
            # RGB - normalize each channel
            img_min = np.percentile(img, 2, axis=(0, 1))
            img_max = np.percentile(img, 98, axis=(0, 1))

        # Apply range
        # Prevent division by zero
        if np.all(img_min == img_max):
            return np.zeros_like(img, dtype=np.uint8)
            
        normalized = np.clip((img - img_min) / (img_max - img_min + 1e-10), 0, 1)

        # Convert to 8-bit
        return (normalized * 255).astype(np.uint8)

    def zoom_image(self):
        """Apply zoom to the image and display it."""
        if self.image is None:
            return

        # Calculate new size
        new_width = int(self.image.width * self.zoom_level)
        new_height = int(self.image.height * self.zoom_level)

        # Ensure minimum size
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # Resize image
        if new_width > 0 and new_height > 0:
            resized_img = self.image.resize((new_width, new_height), Image.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(resized_img)

            # Update canvas
            self.delete("image")
            self.image_id = self.create_image(
                self.winfo_width() // 2 + self.pan_x,
                self.winfo_height() // 2 + self.pan_y,
                anchor="center",
                image=self.photo_image,
                tags=("image"),
            )
            
    def fit_to_canvas(self):
        """Adjust zoom level to fit the image in the canvas."""
        if self.image is None:
            return
            
        # Reset pan position
        self.pan_x = 0
        self.pan_y = 0
        
        # Get canvas dimensions
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        # Ensure the canvas dimensions are valid
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet properly sized, use default zoom
            self.zoom_level = 1.0
            return
            
        # Calculate zoom factors for width and height
        width_zoom = (canvas_width - 20) / self.image.width  # Add some padding
        height_zoom = (canvas_height - 20) / self.image.height  # Add some padding
        
        # Use the smaller zoom factor to ensure the entire image fits
        self.zoom_level = min(width_zoom, height_zoom)
        
        # Ensure zoom level is within bounds
        self.zoom_level = max(self.min_zoom, min(self.zoom_level, self.max_zoom))
        
        # Apply the zoom
        self.zoom_image()

    def on_button_press(self, event):
        """Handle button press event."""
        self.is_panning = True
        self.start_x = event.x
        self.start_y = event.y

    def on_button_motion(self, event):
        """Handle button motion event."""
        if self.is_panning and self.image_id:
            # Calculate movement
            dx = event.x - self.start_x
            dy = event.y - self.start_y

            # Update position
            self.pan_x += dx
            self.pan_y += dy

            # Move the image
            self.move("image", dx, dy)

            # Update starting position
            self.start_x = event.x
            self.start_y = event.y

    def on_button_release(self, event):
        """Handle button release event."""
        self.is_panning = False

    def on_mousewheel(self, event):
        """Handle mousewheel event for zooming."""
        if self.image is None:
            return

        # Determine zoom direction
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            # Zoom in
            zoom_factor = 1.1
        elif event.num == 5 or (hasattr(event, "delta") and event.delta < 0):
            # Zoom out
            zoom_factor = 0.9
        else:
            return

        # Apply zoom limits
        new_zoom = self.zoom_level * zoom_factor
        if new_zoom < self.min_zoom or new_zoom > self.max_zoom:
            return

        # Save old zoom for relative positioning
        old_zoom = self.zoom_level

        # Update zoom level
        self.zoom_level = new_zoom

        # Update image with new zoom
        self.zoom_image()


class FullImageExplorer:
    """Main application for exploring full SAR-Optical satellite images."""

    def __init__(self, root, data_dir, split="train", output_file="image_labels.txt"):
        self.root = root
        self.data_dir = Path(data_dir)
        self.split = split
        self.output_file = output_file

        # Set window title and size
        root.title(f"Full Satellite Image Explorer - {split}")
        root.geometry("1400x900")

        # Initialize image loader
        self.image_loader = SatelliteImageLoader(data_dir)

        # Find all available images
        self.load_image_ids()

        # Current image index
        self.current_index = 0
        
        # Status variable for keyboard actions
        self.status_var = StringVar()
        self.status_var.set("")

        # Create UI components
        self.create_widgets()

    # Bind keyboard events
        self.root.bind("<KeyPress-n>", self.on_key_next)
        self.root.bind("<KeyPress-c>", self.on_key_c)
        self.root.bind("<KeyPress-s>", self.on_key_s)
        
        # Initial image loading will be done via main() after the GUI is set up

        # Will load the first or selected image after initialization

    def load_image_ids(self):
        """Load all available image IDs from the dataset."""
        try:
            post_event_dir = self.data_dir / self.split / "post-event"
            self.image_ids = sorted(
                [
                    f.name.replace("_post_disaster.tif", "")
                    for f in post_event_dir.glob("*_post_disaster.tif")
                ]
            )

            print(f"Found {len(self.image_ids)} images in {self.split} split")
        except Exception as e:
            print(f"Error loading image IDs: {e}")
            self.image_ids = []

    def create_widgets(self):
        """Create the UI widgets."""
        # Main frame
        main_frame = Frame(self.root, bg="#333333")
        main_frame.pack(fill=BOTH, expand=True)

        # Top control panel
        control_panel = Frame(main_frame, bg="#333333")
        control_panel.pack(fill=X, side=TOP, padx=10, pady=10)

        # Image navigation
        nav_frame = Frame(control_panel, bg="#333333")
        nav_frame.pack(side=LEFT)

        prev_btn = Button(
            nav_frame, text="< Previous", command=self.load_previous_image
        )
        prev_btn.pack(side=LEFT, padx=5)

        # Image selector
        self.image_var = StringVar()
        if self.image_ids:
            self.image_var.set(self.image_ids[0])

        image_selector = OptionMenu(
            nav_frame, self.image_var, *self.image_ids, command=self.on_image_select
        )
        image_selector.config(width=20)
        image_selector.pack(side=LEFT, padx=5)

        next_btn = Button(nav_frame, text="Next >", command=self.load_next_image)
        next_btn.pack(side=LEFT, padx=5)

        # Image index display
        self.index_var = StringVar()
        self.update_index_display()
        index_label = Label(
            nav_frame, textvariable=self.index_var, bg="#333333", fg="white"
        )
        index_label.pack(side=LEFT, padx=20)

        # Status display for keyboard actions
        status_label = Label(
            nav_frame, textvariable=self.status_var, bg="#333333", fg="#FFFF00"
        )
        status_label.pack(side=LEFT, padx=20)

        # Display controls
        display_frame = Frame(control_panel, bg="#333333")
        display_frame.pack(side=RIGHT)

        # Contrast controls for pre-event image
        pre_contrast_frame = Frame(display_frame, bg="#333333")
        pre_contrast_frame.pack(side=LEFT, padx=20)

        Label(
            pre_contrast_frame, text="Pre-event Contrast:", bg="#333333", fg="white"
        ).pack(anchor="w")
        self.pre_min_scale = Scale(
            pre_contrast_frame,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            label="Min %",
            command=self.update_pre_contrast,
            bg="#333333",
            fg="white",
        )
        self.pre_min_scale.set(2)
        self.pre_min_scale.pack(fill=X)

        self.pre_max_scale = Scale(
            pre_contrast_frame,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            label="Max %",
            command=self.update_pre_contrast,
            bg="#333333",
            fg="white",
        )
        self.pre_max_scale.set(98)
        self.pre_max_scale.pack(fill=X)

        # Contrast controls for post-event image
        post_contrast_frame = Frame(display_frame, bg="#333333")
        post_contrast_frame.pack(side=LEFT, padx=20)

        Label(
            post_contrast_frame, text="Post-event Contrast:", bg="#333333", fg="white"
        ).pack(anchor="w")
        self.post_min_scale = Scale(
            post_contrast_frame,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            label="Min %",
            command=self.update_post_contrast,
            bg="#333333",
            fg="white",
        )
        self.post_min_scale.set(2)
        self.post_min_scale.pack(fill=X)

        self.post_max_scale = Scale(
            post_contrast_frame,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            label="Max %",
            command=self.update_post_contrast,
            bg="#333333",
            fg="white",
        )
        self.post_max_scale.set(98)
        self.post_max_scale.pack(fill=X)

        # Image display area (3 panels)
        image_area = Frame(main_frame, bg="#222222")
        image_area.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Configure grid for equal sizing
        image_area.columnconfigure(0, weight=1)
        image_area.columnconfigure(1, weight=1)
        image_area.columnconfigure(2, weight=1)
        image_area.rowconfigure(0, weight=0)
        image_area.rowconfigure(1, weight=1)

        # Labels for images
        Label(image_area, text="Pre-event (Optical)", bg="#222222", fg="white").grid(
            row=0, column=0, sticky="ew"
        )
        Label(image_area, text="Post-event (SAR)", bg="#222222", fg="white").grid(
            row=0, column=1, sticky="ew"
        )
        Label(image_area, text="Damage Map", bg="#222222", fg="white").grid(
            row=0, column=2, sticky="ew"
        )

        # Image canvases
        canvas_height = 600
        canvas_width = 400

        self.pre_canvas = ImageCanvas(
            image_area,
            width=canvas_width,
            height=canvas_height,
            bg="#111111",
            highlightthickness=0,
        )
        self.pre_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.post_canvas = ImageCanvas(
            image_area,
            width=canvas_width,
            height=canvas_height,
            bg="#111111",
            highlightthickness=0,
        )
        self.post_canvas.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.target_canvas = ImageCanvas(
            image_area,
            width=canvas_width,
            height=canvas_height,
            bg="#111111",
            highlightthickness=0,
        )
        self.target_canvas.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        # Status and info panel
        info_frame = Frame(main_frame, bg="#333333", height=100)
        info_frame.pack(fill=X, side=BOTTOM, padx=10, pady=10)

        # Image information display
        self.info_var = StringVar()
        info_label = Label(
            info_frame,
            textvariable=self.info_var,
            bg="#333333",
            fg="white",
            justify=LEFT,
            anchor="w",
        )
        info_label.pack(fill=X, padx=10, pady=5)

                    # Navigation instruction
        instruction_text = (
            "Pan: Click and drag | Zoom: Mouse wheel | "
            "Navigate: Previous/Next buttons or dropdown | "
            "Keyboard: 'n' for next, 'c' to mark as 'c', 's' to mark as 's' | "
            "Images auto-fit to canvas"
        )
        instruction_label = Label(
            info_frame, text=instruction_text, bg="#333333", fg="#AAAAAA", anchor="w"
        )
        instruction_label.pack(fill=X, padx=10, pady=5)

        # Add legend for damage colors
        legend_frame = Frame(info_frame, bg="#333333")
        legend_frame.pack(side=RIGHT, padx=10)

        Label(legend_frame, text="Damage Levels:", bg="#333333", fg="white").pack(
            side=LEFT
        )

        damage_levels = [
            ("0: None", "black", "white"),
            ("1: Minor", "green", "white"),
            ("2: Moderate", "yellow", "black"),
            ("3: Major", "orange", "black"),
            ("4: Destroyed", "red", "white"),
        ]

        for text, bg, fg in damage_levels:
            level_label = Label(legend_frame, text=text, bg=bg, fg=fg, padx=5)
            level_label.pack(side=LEFT, padx=2)

    def update_index_display(self):
        """Update the image index display."""
        if not self.image_ids:
            self.index_var.set("No images found")
            return

        self.index_var.set(f"Image {self.current_index + 1} of {len(self.image_ids)}")

    def prompt_for_starting_image(self):
        """Show a dialog to ask the user which image to start from."""
        if not self.image_ids:
            messagebox.showinfo("No Images", "No images found in the dataset.")
            return
            
        # Ask for starting image index or image ID
        prompt = "Enter the index (1-{}) or image ID to start from:".format(len(self.image_ids))
        response = simpledialog.askstring("Starting Image", prompt)
        
        if response:
            try:
                # Try to interpret as index
                index = int(response) - 1  # Convert to 0-based index
                if 0 <= index < len(self.image_ids):
                    self.current_index = index
                    self.load_current_image()
                else:
                    # Out of range
                    messagebox.showwarning("Invalid Index", f"Index must be between 1 and {len(self.image_ids)}")
                    # Load first image as fallback
                    self.current_index = 0
                    self.load_current_image()
            except ValueError:
                # Try to interpret as image ID
                if response in self.image_ids:
                    self.current_index = self.image_ids.index(response)
                    self.load_current_image()
                else:
                    # Not found
                    messagebox.showwarning("Invalid Image ID", f"Image ID '{response}' not found")
                    # Load first image as fallback
                    self.current_index = 0
                    self.load_current_image()
        else:
            # If canceled or empty, start with first image
            self.current_index = 0
            self.load_current_image()
    
    def load_current_image(self):
        """Load the current image."""
        if not self.image_ids:
            return

        # Get image ID
        image_id = self.image_ids[self.current_index]

        # Update UI
        self.image_var.set(image_id)
        self.update_index_display()

        # Check cache first
        cached = self.image_loader.get_cached(image_id)
        if cached:
            self.update_canvases(
                cached["pre"], cached["post"], cached["target"], cached["info"]
            )
            return

        # Load the image pair
        self.info_var.set(f"Loading image {image_id}...")
        self.image_loader.load_image_pair(image_id, self.split, self.update_canvases)

    def update_canvases(self, pre_img, post_img, target_img, info):
        """Update the canvases with the loaded images."""
        # Update pre-event canvas
        self.pre_canvas.set_image(pre_img, "Pre-event Optical")

        # Update post-event canvas
        self.post_canvas.set_image(post_img, "Post-event SAR")

        # Update target canvas with enhanced visualization
        if target_img is not None:
            # Visualize damage levels with colors
            target_viz = self.visualize_damage(target_img)
            self.target_canvas.set_image(target_viz, "Damage Map")
        else:
            self.target_canvas.set_image(None)

        # Update info
        if info:
            info_text = (
                f"Image ID: {self.image_ids[self.current_index]}\n"
                f"Size: {info['width']} x {info['height']} pixels\n"
                f"CRS: {info['crs']}, Bands: {info['bands']}, Type: {info['dtype']}"
            )
            self.info_var.set(info_text)
        else:
            self.info_var.set("Error loading image information")

    def visualize_damage(self, damage_img):
        """Create a color visualization of damage levels."""
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
        h, w = damage_img.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Fill with colors based on damage values
        for i in range(5):  # 5 damage levels
            mask = damage_img == i
            if np.any(mask):
                rgb[mask] = colors[i]

        return rgb

    def load_next_image(self):
        """Load the next image."""
        if not self.image_ids:
            return

        self.current_index = (self.current_index + 1) % len(self.image_ids)
        self.load_current_image()

    def load_previous_image(self):
        """Load the previous image."""
        if not self.image_ids:
            return

        self.current_index = (self.current_index - 1) % len(self.image_ids)
        self.load_current_image()

    def on_image_select(self, selection):
        """Handle image selection from dropdown."""
        try:
            self.current_index = self.image_ids.index(selection)
            self.load_current_image()
        except ValueError:
            print(f"Image {selection} not found in image list")

    def update_pre_contrast(self, *args):
        """Update the contrast of the pre-event image."""
        min_val = self.pre_min_scale.get()
        max_val = self.pre_max_scale.get()
        self.pre_canvas.update_display(min_val, max_val)

    def update_post_contrast(self, *args):
        """Update the contrast of the post-event image."""
        min_val = self.post_min_scale.get()
        max_val = self.post_max_scale.get()
        self.post_canvas.update_display(min_val, max_val)

    def on_key_next(self, event):
        """Handle 'n' key press (next image)."""
        self.status_var.set(f"Next image (#{self.current_index + 2})")
        self.load_next_image()

    def on_key_c(self, event):
        """Handle 'c' key press."""
        if not self.image_ids:
            return

        # Get current image ID
        image_id = self.image_ids[self.current_index]
        
        # Write to file
        self.write_to_file(image_id, "c")
        
        # Show status
        self.status_var.set(f"Marked '{image_id}' (#{self.current_index + 1}) as 'c'")
        
        # Load next image
        self.load_next_image()

    def on_key_s(self, event):
        """Handle 's' key press."""
        if not self.image_ids:
            return

        # Get current image ID
        image_id = self.image_ids[self.current_index]
        
        # Write to file
        self.write_to_file(image_id, "s")
        
        # Show status
        self.status_var.set(f"Marked '{image_id}' (#{self.current_index + 1}) as 's'")
        
        # Load next image
        self.load_next_image()

    def write_to_file(self, image_id, label):
        """Write the image ID, index, and label to the output file."""
        try:
            # Create the directory if it doesn't exist
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get the index (1-based for better readability)
            image_index = self.current_index + 1
            
            # Write to file (append mode)
            with open(output_path, "a") as f:
                f.write(f"{image_id},{image_index},{label}\n")
                
            print(f"Added '{image_id},{image_index},{label}' to {self.output_file}")
        except Exception as e:
            print(f"Error writing to file: {e}")
            self.status_var.set(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Full image explorer for SAR-Optical change detection dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/dfc25_track2_trainval",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="image_labels.txt",
        help="Output file for image labels",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Starting image index (1-based)",
    )
    parser.add_argument(
        "--start_id",
        type=str,
        default=None,
        help="Starting image ID",
    )
    args = parser.parse_args()

    # Create the Tkinter application
    root = Tk()
    app = FullImageExplorer(root, args.data_dir, args.split, args.output_file)
    
    # After the GUI is fully set up, process any starting parameters
    def after_setup():
        # Determine if we should use command line args or prompt
        use_prompt = True
        
        # If a starting index or ID was specified via command line
        if args.start_index is not None and args.start_index > 0:
            index = args.start_index - 1  # Convert to 0-based
            if 0 <= index < len(app.image_ids):
                app.current_index = index
                app.load_current_image()
                use_prompt = False  # Skip the prompt
        elif args.start_id is not None:
            if args.start_id in app.image_ids:
                app.current_index = app.image_ids.index(args.start_id)
                app.load_current_image()
                use_prompt = False  # Skip the prompt
        
        # Only show the prompt if no valid command-line args were provided
        if use_prompt:
            app.prompt_for_starting_image()
        
    # Schedule this to run after main window is up
    root.after(100, after_setup)
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()