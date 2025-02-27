from fastapi import FastAPI
from src.v1.art_matching import ArtMatching
from src.v1.art_image import ArtImage
from pathlib import Path
from pydantic import BaseModel
from typing import List
from urllib.parse import urlparse

from gui.gui_logic import GuiLogic



app = FastAPI()
art_matching = ArtMatching()
gui_logic = GuiLogic()


class SimilarityRequest(BaseModel):
    liked_ids: List[int]
    disliked_ids: List[int]


@app.post('/insert_arts')
def insert_arts(images: list[ArtImage]) -> bool:
    # should return bool
    for image in images:
        image.url = str(Path(image.url).resolve())  # Normalize for all OS
    return art_matching.insert_arts(images)


@app.post('/del_image')
def del_image(image_id) -> bool:
    return art_matching.del_image(image_id)


@app.post('/get_similar_arts')
def get_similar_arts(data: SimilarityRequest):
    # should return list of ids
    return art_matching.get_similar_arts(data)


@app.post('/get_arts_data')
def get_image_data(images_id):
    return art_matching.get_arts_data(images_id)


@app.post('/local_insert')
def local_insert():
    gui_logic.insert()
