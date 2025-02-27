from src.v1.art_image import ArtImage
from src.db.db_manager import DBManager

class ConfigDB:
    img_embed_dim = 768
    prompt_embed_dim = 2
    collection_name = 'v1'
    db_name = 'db.db'
    host = "localhost"
    port = 19530


class DBApi:

    def __init__(self, init: bool):
        self.config_db = ConfigDB()
        self.db_manager = DBManager(config=self.config_db, init=init)

    def insert_arts(self, arts: list[ArtImage]):
        # process ArtImage to match the db
        batch = []
        for art in arts:
            batch.append({
                "id": art.id,
                "url": art.url,
                "img_name": art.img_name,
                "img_embedding": art.img_embeddings,
                "created_at": str(art.created_at),
                "prompt": art.prompt or "",
                "prompt_embedding": art.prompt_embeddings or [0, 0],
                "artist": art.artist or "",
                "size": str(art.size),
                "extracted_features": str(art.extracted_features) or "",
                "tags": ','.join(art.tags or ''),
            })

        self.db_manager.set(batch)

    def index(self):
        self.db_manager.index()

    def delete_arts(self, arts_ids):
        pass

    def get_arts_data(self, arts_ids: list):
        pass

    def get_similar_arts(self, arts: list[ArtImage] | list[int], top_k: int):
        """or list of art object, or list pf ids"""
        if isinstance(arts[0], ArtImage):
            arts_embeddings = [arts.img_embeddings for art in arts]
            arts = self.db_manager.get_similarity_by_embeddings(arts_embeddings)
            return arts
        elif isinstance(arts[0], int):
            arts = self.db_manager.get_similarity_by_ids(arts)
            return arts

        