from pydantic import BaseModel
from datetime import datetime
import torch


class SearchImage(BaseModel):
    room: str | None = None
    user_data: str | None = None
    liked_images: list | None = None  # ArtImages or image_id.
    disliked_images: list | None = None
    created_at: datetime


class Volumes(BaseModel):
    model_folder: str = 'weights'
    db_folder: str = 'db_folder'


