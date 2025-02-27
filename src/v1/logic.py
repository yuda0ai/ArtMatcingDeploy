from src.v1.art_image import GenArtImages, ArtImage
from src.db.db_api import DBApi
from src.v1.embed_model import EmbedModel, ClipEmbed, ConfigClip
from typing import Literal
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class Logic:

    def __init__(self):
        self.db_api = DBApi(init=True)
        self.embed_model: EmbedModel = ClipEmbed(ConfigClip())
        self.batch_size = 4
        self.gen_art_images = GenArtImages(self.embed_model, self.batch_size)


    def insert_art_images(self, imgs: list[ArtImage]):
        gen_embed = self.gen_art_images.gen_object(imgs)
        for batch_obj in tqdm(gen_embed, desc=f'{len(imgs) / self.batch_size}'):
            self.db_api.insert_arts(batch_obj)
        return True
    

    def borda_count(self, rankings, dislikes=None):
        scores = defaultdict(int)
        max_rank = max(len(r) for r in rankings) if rankings else 0  # Maximum rank across all lists
        max_dislike_rank = max(len(d) for d in dislikes) if dislikes else 0  # Max rank for dislikes

        for ranking in rankings:
            for rank, image_id in enumerate(ranking):
                scores[image_id] += (max_rank - rank)  # Higher rank -> more points

        if dislikes:
            for dislike_ranking in dislikes:
                for rank, image_id in enumerate(dislike_ranking):
                    scores[image_id] -= (max_dislike_rank - rank)


        sorted_images = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [image_id for image_id, _ in sorted_images]


    def get_similar_arts(self, data, top_n=6, top_k=1000):
        liked_arts_ids, disliked_arts_ids = data.liked_ids, data.disliked_ids
        set_ids = set(liked_arts_ids + disliked_arts_ids)

        liked_similarity_lists = []
        disliked_similarity_lists = []

        # Fetch embeddings for liked arts
        if liked_arts_ids:
            liked_similarity_lists = self.db_api.get_similar_arts(liked_arts_ids, top_k=top_k)
        
        print(len(liked_arts_ids))

        # Fetch embeddings for disliked arts
        if disliked_arts_ids:
            disliked_similarity_lists = self.db_api.get_similar_arts(disliked_arts_ids, top_k=top_k)

        print(len(liked_similarity_lists))

        # Apply ranking algorithm
        similarity_list = self.borda_count(liked_similarity_lists, disliked_similarity_lists)

        # Exclude liked and disliked items
        similarity_list = [similar_id for similar_id in similarity_list if similar_id not in set_ids][:top_n]

        print(f"similar arts: {similarity_list}")

        return similarity_list

    
    


