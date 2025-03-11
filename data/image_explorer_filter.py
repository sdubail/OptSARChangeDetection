#!/usr/bin/env python
"""
Full image explorer for SAR-Optical change detection dataset.
Allows browsing and analysis of full satellite images with zoom and pan capabilities.
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
)

import numpy as np
import rasterio
from PIL import Image, ImageTk
from scipy.ndimage import generic_filter

# Set this to handle larger images
Image.MAX_IMAGE_PIXELS = None


def lee_filter(img, size=7, strength=1.0):
    """
    Apply Lee filter to reduce speckle noise in SAR imagery.

    Args:
        img: Input image (2D numpy array)
        size: Filter window size (odd number)
        strength: Filter strength factor (higher = more filtering)

    Returns:
        Filtered image
    """
    # Calculate statistics in local windows
    img_mean = generic_filter(img, np.mean, size=size)
    img_sqr_mean = generic_filter(np.square(img), np.mean, size=size)
    img_variance = img_sqr_mean - np.square(img_mean)

    # Calculate overall variance (noise variance)
    overall_variance = np.var(img) * strength

    # Calculate the weight (adaptive factor)
    weight = img_variance / (img_variance + overall_variance + 1e-10)

    # Apply the filter
    return img_mean + weight * (img - img_mean)


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
                callback(None, None, None, None, None)
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

            # Create an initial copy for filtered image (don't apply filter yet)
            post_img_filtered = post_img.copy()

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
                {
                    "pre": pre_img,
                    "post": post_img,
                    "post_filtered": post_img_filtered,
                    "target": target_img,
                    "info": info,
                },
            )

            # Call the callback with the loaded images
            if callback:
                callback(pre_img, post_img, post_img_filtered, target_img, info)

        except Exception as e:
            print(f"Error in load thread: {e}")
            if callback:
                callback(None, None, None, None, None)

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
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.start_x = 0
        self.start_y = 0

        # Image data
        self.numpy_image = None  # Original numpy image
        self.display_range = (0, 255)  # Range for display scaling

        # Setup interactions
        self.bind("<ButtonPress-1>", self.on_button_press)
        self.bind("<B1-Motion>", self.on_button_motion)
        self.bind("<ButtonRelease-1>", self.on_button_release)
        self.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down

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

        # Reset zoom and pan
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0

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
            img_min = np.percentile(img, display_min)
            img_max = np.percentile(img, display_max)
        else:
            # RGB - normalize each channel
            img_min = np.percentile(img, display_min, axis=(0, 1))
            img_max = np.percentile(img, display_max, axis=(0, 1))

        # Apply range
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

    def __init__(self, root, data_dir, split="train"):
        self.root = root
        self.data_dir = Path(data_dir)
        self.split = split

        # Set window title and size
        root.title(f"Full Satellite Image Explorer - {split}")
        root.geometry("1400x900")

        # Initialize image loader
        self.image_loader = SatelliteImageLoader(data_dir)

        # Find all available images
        self.load_image_ids()

        # Current image index
        self.current_index = 0

        # Create UI components
        self.create_widgets()

        # Load first image
        if self.image_ids:
            self.load_current_image()

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

        # Lee filter controls
        filter_frame = Frame(display_frame, bg="#333333")
        filter_frame.pack(side=LEFT, padx=20)

        Label(filter_frame, text="Lee Filter:", bg="#333333", fg="white").pack(
            anchor="w"
        )

        self.filter_size_var = StringVar(value="7")
        filter_sizes = ["3", "5", "7", "9", "11"]
        filter_size_menu = OptionMenu(
            filter_frame,
            self.filter_size_var,
            *filter_sizes,
            command=self.update_filter,
        )
        filter_size_menu.pack(fill=X)

        # Add filter strength control
        Label(filter_frame, text="Filter Strength:", bg="#333333", fg="white").pack(
            anchor="w"
        )
        self.filter_strength_scale = Scale(
            filter_frame,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            orient=HORIZONTAL,
            command=self.update_filter,
            bg="#333333",
            fg="white",
        )
        self.filter_strength_scale.set(1.0)
        self.filter_strength_scale.pack(fill=X)

        # Apply filter button
        apply_filter_btn = Button(
            filter_frame, text="Apply Filter", command=self.update_filter
        )
        apply_filter_btn.pack(fill=X, pady=5)

        # Image display area (4 panels)
        image_area = Frame(main_frame, bg="#222222")
        image_area.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Configure grid for equal sizing
        image_area.columnconfigure(0, weight=1)
        image_area.columnconfigure(1, weight=1)
        image_area.columnconfigure(2, weight=1)
        image_area.columnconfigure(3, weight=1)
        image_area.rowconfigure(0, weight=0)
        image_area.rowconfigure(1, weight=1)

        # Labels for images
        Label(image_area, text="Pre-event (Optical)", bg="#222222", fg="white").grid(
            row=0, column=0, sticky="ew"
        )
        Label(image_area, text="Post-event (SAR)", bg="#222222", fg="white").grid(
            row=0, column=1, sticky="ew"
        )
        Label(
            image_area, text="Post-event (SAR+Lee Filter)", bg="#222222", fg="white"
        ).grid(row=0, column=2, sticky="ew")
        Label(image_area, text="Damage Map", bg="#222222", fg="white").grid(
            row=0, column=3, sticky="ew"
        )

        # Image canvases
        canvas_height = 600
        canvas_width = 300  # Reduced width to fit 4 panels

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

        self.post_filtered_canvas = ImageCanvas(
            image_area,
            width=canvas_width,
            height=canvas_height,
            bg="#111111",
            highlightthickness=0,
        )
        self.post_filtered_canvas.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        self.target_canvas = ImageCanvas(
            image_area,
            width=canvas_width,
            height=canvas_height,
            bg="#111111",
            highlightthickness=0,
        )
        self.target_canvas.grid(row=1, column=3, sticky="nsew", padx=5, pady=5)

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
            "Navigate: Previous/Next buttons or dropdown"
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
                cached["pre"],
                cached["post"],
                cached["post_filtered"],
                cached["target"],
                cached["info"],
            )
            return

        # Load the image pair
        self.info_var.set(f"Loading image {image_id}...")
        self.image_loader.load_image_pair(image_id, self.split, self.update_canvases)

    def update_filter(self, *args):
        """Update the Lee filter parameters and reapply filter."""
        if not self.image_ids:
            return

        # Show a message on the canvas indicating filtering is in progress
        self.post_filtered_canvas.delete("all")
        self.post_filtered_canvas.create_text(
            self.post_filtered_canvas.winfo_width() // 2,
            self.post_filtered_canvas.winfo_height() // 2,
            text="Applying filter...",
            fill="white",
        )
        self.root.update()  # Force UI update

        # Start filtering in a separate thread to avoid freezing the UI
        threading.Thread(target=self._apply_filter).start()

    def _apply_filter(self):
        """Apply the Lee filter in a background thread."""
        try:
            # Get current image ID
            image_id = self.image_ids[self.current_index]

            # Check cache
            cached = self.image_loader.get_cached(image_id)
            if cached and "post" in cached:
                # Get filter parameters
                filter_size = int(self.filter_size_var.get())
                filter_strength = float(self.filter_strength_scale.get())

                # Apply filter to SAR image
                post_img = cached["post"]
                post_img_filtered = post_img.copy()

                # Apply Lee filter
                if post_img.ndim == 2:
                    post_img_filtered = lee_filter(
                        post_img, size=filter_size, strength=filter_strength
                    )
                elif post_img.ndim == 3:
                    for i in range(post_img.shape[2]):
                        post_img_filtered[:, :, i] = lee_filter(
                            post_img[:, :, i],
                            size=filter_size,
                            strength=filter_strength,
                        )

                # Update cache
                cached["post_filtered"] = post_img_filtered

                # Update canvas (must be done on the main thread)
                self.root.after(
                    0,
                    lambda: self.post_filtered_canvas.set_image(
                        post_img_filtered,
                        f"Post-event SAR (Lee {filter_size}×{filter_size}, S={filter_strength})",
                    ),
                )
        except Exception as e:
            print(f"Error applying filter: {e}")
            # Update UI to show error
            self.root.after(0, lambda: self.post_filtered_canvas.delete("all"))
            self.root.after(
                0,
                lambda: self.post_filtered_canvas.create_text(
                    self.post_filtered_canvas.winfo_width() // 2,
                    self.post_filtered_canvas.winfo_height() // 2,
                    text="Error applying filter",
                    fill="red",
                ),
            )

    def update_canvases(self, pre_img, post_img, post_img_filtered, target_img, info):
        """Update the canvases with the loaded images."""
        # Update pre-event canvas
        self.pre_canvas.set_image(pre_img, "Pre-event Optical")

        # Update post-event canvas
        self.post_canvas.set_image(post_img, "Post-event SAR")

        # Update filtered SAR canvas
        filter_size = int(self.filter_size_var.get())
        filter_strength = float(self.filter_strength_scale.get())
        self.post_filtered_canvas.set_image(
            post_img_filtered,
            f"Post-event SAR (Lee {filter_size}×{filter_size}, S={filter_strength})",
        )

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
    args = parser.parse_args()

    # Create the Tkinter application
    root = Tk()
    app = FullImageExplorer(root, args.data_dir, args.split)

    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()
