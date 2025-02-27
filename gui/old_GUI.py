import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps
import math
import random
from gui.gui_logic import Logic


class ConfigGui:

    def __init__(self):
        self.images_folder="data/downloaded_images"
        self.image_size=(150, 150)
        self.padding_color=(100, 100, 100)

if "logic_instance" not in st.session_state:
    st.session_state.logic_instance = Logic()


class GUI:
    def __init__(self):
        images_folder="data/downloaded_images"
        image_size=(150, 150)
        padding_color=(100, 100, 100)
        self.config = {
            'images_folder': images_folder,
            'image_size': image_size,
            'padding_color': padding_color,
            'images_per_row': 4
        }

        # Initialize session states if needed
        if "selected_images" not in st.session_state:
            st.session_state.selected_images = set()
        if "current_images" not in st.session_state:
            st.session_state.current_images = []
        if "predicted_images" not in st.session_state:
            st.session_state.predicted_images = []
        if "all_available_images" not in st.session_state:
            st.session_state.all_available_images = list(Path(self.config['images_folder']).glob("*.jpg"))
        if "user_description" not in st.session_state:
            st.session_state.user_description = ""

    def process_image(self, image_path):
        """Standardize image size with padding"""
        img = Image.open(image_path).convert("RGB")
        return ImageOps.pad(img, self.config['image_size'], color=self.config['padding_color'])

    def load_images(self, max_images=20):
        """Load images from the specified folder"""
        if not st.session_state.current_images:
            st.session_state.current_images = random.sample(
                st.session_state.all_available_images,
                min(max_images, len(st.session_state.all_available_images))
            )
        return st.session_state.current_images

    def sample_new_images(self, max_images=20):
        """Sample a new set of images"""
        available_images = [img for img in st.session_state.all_available_images
                            if str(img) not in st.session_state.selected_images]

        if available_images:
            st.session_state.current_images = random.sample(
                available_images,
                min(max_images, len(available_images))
            )

    def create_image_grid(self, images):
        """Create a grid layout of images"""
        total_images = len(images)
        cols_per_row = self.config['images_per_row']
        rows = math.ceil(total_images / cols_per_row)

        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                img_idx = row * cols_per_row + col_idx
                if img_idx < total_images:
                    with cols[col_idx]:
                        self.display_selectable_image(images[img_idx])

    def display_selectable_image(self, image_path):
        """Display a single image with selection button"""
        img = self.process_image(image_path)
        str_path = str(image_path)

        # Display the image
        st.image(img, use_column_width=True)

        # Create button with dynamic color based on selection state
        button_color = "primary" if str_path in st.session_state.selected_images else "secondary"
        if st.button("Select", key=f"btn_{str_path}", type=button_color):
            if str_path in st.session_state.selected_images:
                st.session_state.selected_images.remove(str_path)
            else:
                st.session_state.selected_images.add(str_path)
            st.rerun()

    def predict_images(self):
        """
        Prediction logic with user description.
        """
        if len(st.session_state.selected_images) >= 1 or st.session_state.user_description:
            filenames = list(st.session_state.selected_images)
            user_desc = st.session_state.user_description if st.session_state.user_description else None
            return st.session_state.logic_instance.predict_images(
                filenames,
                user_desc=user_desc
            )
        else:
            return []

    def display_predicted_images(self):
        """Display the predicted/ordered images"""
        if st.session_state.predicted_images:
            st.subheader("Predicted Images")
            cols = st.columns(len(list(set(st.session_state.predicted_images))))
            for idx, img_path in enumerate(list(set(st.session_state.predicted_images))):
                with cols[idx]:
                    img = self.process_image(img_path)
                    st.image(img, use_column_width=True)

    def display_selected_images(self):
        """Display the selected images"""
        if st.session_state.selected_images:
            st.subheader("Selected Images")
            cols = st.columns(max(len(list(st.session_state.selected_images)), 4))
            for idx, img_path in enumerate(list(st.session_state.selected_images)):
                with cols[idx]:
                    img = self.process_image(img_path)
                    st.image(img, use_column_width=True)

    def run(self):
        """Main method to run the interface"""
        st.title("Image Selection Interface")

        # Add user description input at the top
        st.text_area(
            "Describe what you're looking for:",
            key="user_description",
            height=100,
            help="Enter a description to help guide the image prediction"
        )

        # Add control buttons at the top
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

        # Display selected images count
        st.write(f"Selected images: {len(st.session_state.selected_images)}")

        with col1:
            if st.button("Clear All"):
                st.session_state.selected_images.clear()
                st.session_state.predicted_images = []
                st.rerun()

        with col2:
            if st.button("Predict"):
                st.session_state.predicted_images = self.predict_images()
                st.rerun()

        with col3:
            if st.button("Sample"):
                self.sample_new_images()
                st.rerun()

        # Load and display main image grid
        images = self.load_images()
        self.create_image_grid(images)

        self.display_selected_images()

        # Display predicted images
        self.display_predicted_images()

