import numpy as np
from pymilvus import connections
from pymilvus import MilvusClient, DataType, Collection, FieldSchema, CollectionSchema, utility


class DBManager:

    def __init__(self, config, init: bool = False):
        self.config = config
        self.img_embed_dim = config.img_embed_dim
        self.prompt_embed_dim = config.prompt_embed_dim
        self.collection_name = config.collection_name
        self.conn = connections.connect(alias="default", host=config.host, port=config.port)
        # self.client = MilvusClient(config.db_name)
        if init:
            self._create_collection()
        else:
            self.collection = Collection(name=self.collection_name)
        self.index()


    def _create_collection(self):
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="img_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.img_embed_dim),
            FieldSchema(name="img_name", dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name="prompt", dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name="prompt_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.prompt_embed_dim),
            FieldSchema(name="extracted_features", dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='artist', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='tags', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='size', dtype=DataType.VARCHAR, max_length=64000),
        ]

        schema = CollectionSchema(fields, description="Art Matching Collection")

        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"Collection {self.collection_name} created!")

    def index(self):

        # Create index for the first embedding field
        index_params_1 = {
            "metric_type": "COSINE",  # Inner Product for cosine similarity
            "index_type": "AUTOINDEX",
            "params": {}
        }
        self.collection.create_index(
            field_name="img_embedding",  # Your first embedding field
            index_params=index_params_1
        )

        # Create index for the second embedding field
        index_params_2 = {
            "metric_type": "COSINE",  # Can use different metric type if needed
            "index_type": "AUTOINDEX", 
            "params": {}
        }
        self.collection.create_index(
            field_name="prompt_embedding",  # Your second embedding field
            index_params=index_params_2
        )

        # Load the collection to use both indexes
        self.collection.load()

    def set(self, batch):
        self.collection.insert(data=batch)

    def delete(self, ids):
        self.collection.delete(ids=ids)

    def get_similarity_by_embeddings(self, embeddings: list[list[float]], anns_field="img_embedding", top_k=1000): 
        """
        Return the top_k vectors in the db that are most similar to the given embeddings.
        """
        top_k = top_k if top_k is not None else 1

        # Define search parameters
        search_params = {
            "metric_type": "COSINE",  # Ensure this matches the index metric
            "params": {}  # Empty params for AUTOINDEX
        }

        # Perform the search in Milvus
        results = self.collection.search(
            data=embeddings,  # The query vectors
            anns_field=anns_field,  # The field to search in
            param=search_params,  # The search parameter dictionary
            limit=top_k,  # Number of top results to return
            output_fields=['id']  # Fields to return (e.g., 'id', 'url', etc.)
        )

        # Extract the relevant search results (i.e., IDs of matching vectors)
        output = []
        for result in results:
            output.append([hit.entity.get('id') for hit in result])

        return output

    

    def get_similarity_by_ids(self, ids: list[int], top_k=1000):
        """
        Given a list of IDs, retrieve their embeddings from Milvus and return the top_k most similar vectors' IDs.
        """
        top_k = top_k if top_k is not None else 1

        # Step 1: Retrieve embeddings of the given IDs
        id_list = ", ".join(str(id) for id in ids)
        print(id_list)
        if id_list:
            entities = self.collection.query(
                expr=f"id in [{id_list}]",
                output_fields=["img_embedding"]
            )
        else:
            entities = []
        
        print(entities, 'ent')

        if not entities:
            return []  # If no embeddings are found, return empty list

        # Extract embeddings
        embeddings = [entity["img_embedding"] for entity in entities]

        return self.get_similarity_by_embeddings(embeddings=embeddings, top_k=top_k)