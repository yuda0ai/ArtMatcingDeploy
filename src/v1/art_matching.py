from src.v1.logic import Logic
from src.v1.art_image import ArtImage

class ArtMatching:
    """the main class"""

    def __init__(self):
        self.logic = Logic()

    def insert_arts(self, imgs: list[ArtImage]) -> bool:
        self.logic.insert_art_images(imgs)
        return True

    def del_arts(self, images_id) -> bool:
        # 1. delete from the database
        return True

    def get_similar_arts(self, data):
        # should return list of ids
        arts = self.logic.get_similar_arts(data)
        return arts

    def get_arts_data(self, images_id):
        return True
