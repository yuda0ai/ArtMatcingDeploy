import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps
import math
import random
from gui_logic import GuiLogic

class ConfigGui:
    def __init__(self):
        self.images_folder = "data"
        self.image_size = (150, 150)
        self.padding_color = (100, 100, 100)


class GUI:
    def __init__(self):
        config = ConfigGui()
        images_folder = config.images_folder 
        image_size = config.image_size 
        padding_color = config.padding_color 
        self.config = {
            'images_folder': images_folder,
            'image_size': image_size,
            'padding_color': padding_color,
            'images_per_row': 4
        }

        # Initialize session states if needed
        if "logic_instance" not in st.session_state:
            st.session_state.logic_instance = GuiLogic()
        if "liked_ids" not in st.session_state:
            st.session_state.liked_ids = set()
        if "disliked_ids" not in st.session_state:
            st.session_state.disliked_ids = set()
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
        # Modified to not exclude disliked images
        available_images = [img for img in st.session_state.all_available_images
                            if str(img) not in st.session_state.liked_ids]

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
        """Display a single image with like and dislike buttons"""
        img = self.process_image(image_path)
        str_path = str(image_path)
        img_id = str_path.split("\\")[-1].split(".")[0]  # Extract the image ID from the file name

        # Display the image
        st.image(img, use_container_width=True)

        # Create like/dislike buttons
        col1, col2 = st.columns([1, 1])

        # Like button
        with col1:
            if st.button(f"Like", key=f"like_{img_id}"):
                st.session_state.liked_ids.add(img_id)
                st.session_state.disliked_ids.discard(img_id)
                st.rerun()

        # Dislike button
        with col2:
            if st.button(f"Dislike", key=f"dislike_{img_id}"):
                st.session_state.disliked_ids.add(img_id)
                st.session_state.liked_ids.discard(img_id)
                st.rerun()

    def predict_images(self):
        """Prediction logic with liked and disliked image IDs."""
        if len(st.session_state.liked_ids) > 0 or len(st.session_state.disliked_ids) > 0:
            return st.session_state.logic_instance.similarity(
                list(st.session_state.liked_ids),
                list(st.session_state.disliked_ids)
            )
        else:
            return []

    def display_predicted_images(self):
        """Display the predicted/ordered images"""
        if st.session_state.predicted_images:
            st.subheader("Predicted Images")
            cols = st.columns(len(list(set(st.session_state.predicted_images))))
            for idx, img_id in enumerate(list(set(st.session_state.predicted_images))):
                with cols[idx]:
                    img_path = Path(self.config['images_folder']) / f"{img_id}.jpg"
                    img = self.process_image(img_path)
                    st.image(img, use_container_width=True)

    def display_selected_images(self):
        """Display the liked and disliked images separately"""
        # Display liked images
        if st.session_state.liked_ids:
            st.subheader("Liked Images")
            liked_images = list(st.session_state.liked_ids)
            liked_cols_per_row = min(self.config['images_per_row'], len(liked_images))
            liked_rows = math.ceil(len(liked_images) / liked_cols_per_row)
            
            for row in range(liked_rows):
                cols = st.columns(liked_cols_per_row)
                for col_idx in range(liked_cols_per_row):
                    img_idx = row * liked_cols_per_row + col_idx
                    if img_idx < len(liked_images):
                        with cols[col_idx]:
                            img_id = liked_images[img_idx]
                            img_path = Path(self.config['images_folder']) / f"{img_id}.jpg"
                            img = self.process_image(img_path)
                            st.image(img, use_container_width=True)
        
        # Display disliked images separately
        if st.session_state.disliked_ids:
            st.subheader("Disliked Images")
            disliked_images = list(st.session_state.disliked_ids)
            disliked_cols_per_row = min(self.config['images_per_row'], len(disliked_images))
            disliked_rows = math.ceil(len(disliked_images) / disliked_cols_per_row)
            
            for row in range(disliked_rows):
                cols = st.columns(disliked_cols_per_row)
                for col_idx in range(disliked_cols_per_row):
                    img_idx = row * disliked_cols_per_row + col_idx
                    if img_idx < len(disliked_images):
                        with cols[col_idx]:
                            img_id = disliked_images[img_idx]
                            img_path = Path(self.config['images_folder']) / f"{img_id}.jpg"
                            img = self.process_image(img_path)
                            st.image(img, use_container_width=True)

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

        # Display liked/disliked images count
        st.write(f"Liked images: {len(st.session_state.liked_ids)}")
        st.write(f"Disliked images: {len(st.session_state.disliked_ids)}")

        with col1:
            if st.button("Clear All"):
                st.session_state.liked_ids.clear()
                st.session_state.disliked_ids.clear()
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

        # Display liked and disliked images separately
        self.display_selected_images()

        # Display predicted images
        self.display_predicted_images()

if __name__ == '__main__':
    selector = GUI()
    selector.run()