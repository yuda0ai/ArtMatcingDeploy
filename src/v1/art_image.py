from pydantic import BaseModel
from datetime import datetime
from src.v1.embed_model import EmbedModel
from typing import List, Dict, Optional, Tuple



class ArtImage(BaseModel):
    id: int
    url: str
    img_name: str
    img_embeddings: Optional[List[float]] = None
    created_at: datetime = datetime.now()  # Automatically set the created_at field
    prompt: Optional[str] = None
    prompt_embeddings: Optional[List[float]] = None
    artist: Optional[str] = None
    size: Tuple[int, int]
    extracted_features: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None


class GenArtImages:

    def __init__(self, model: EmbedModel, batch_size: int):
        self.model = model
        self.batch_size = batch_size

    
    def gen_object(self, imgs: list[ArtImage]):
        """
        each img is object (dict / pydantic) that we get from the user
        """

        for i in range(0, len(imgs), self.batch_size):

            imgs_batch = imgs[i: i + self.batch_size]
            imgs_embed_batch, prompt_batch , prompts_embed_batch = self.get_batch_embeddings([img.url for img in imgs_batch])
            
            for i, img in enumerate(imgs_batch):
                img.img_embeddings = imgs_embed_batch[i]
                img.prompt = prompt_batch[i]
                img.prompt_embeddings = prompts_embed_batch[i]
            
            yield imgs_batch


    def get_batch_embeddings(self, imgs_batch: list[str]):
        imgs_embed_batch = self.model.predict_imgs(imgs_batch)
        prompt_batch = self.get_batch_prompts(imgs_batch)
        prompts_embed_batch = self.get_batch_prompts_embeds(prompt_batch)
        return imgs_embed_batch, prompt_batch , prompts_embed_batch


    def get_batch_prompts(self, images):
        return [None for _ in range(len(images))]

    def get_batch_prompts_embeds(self, prompts):
        return [None for _ in range(len(prompts))]