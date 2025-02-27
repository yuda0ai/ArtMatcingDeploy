import os
import requests

class GuiLogic:

    def __init__(self, insert_endpoint='http://localhost:8000/insert_arts', similarity_endpoint='http://localhost:8000/get_similar_arts', images_dir='data'):
        self.image_dir = images_dir
        self.data = []
        self.insert_endpoint = insert_endpoint
        self.similarity_endpoint = similarity_endpoint

        for filename in os.listdir(self.image_dir):
            if filename.endswith(".jpg"):  # Ensure it's an image file
                img_id = os.path.splitext(filename)[0]  # Extract ID from filename (without extension)
                image_path = os.path.join(self.image_dir, filename)

                self.data.append({
                    "id": int(img_id),  # Convert ID to integer
                    "url": image_path,  # FastAPI expects 'url' to be the filename
                    "img_name": "string",  # Placeholder name, update as needed
                    "size": [0, 0]
                })


    def insert(self):
        try:
            response = requests.post(self.insert_endpoint, json=self.data)
            if response.status_code == 200:
                print("Data successfully sent to FastAPI")
            else:
                print(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Failed to send data: {e}")


    def similarity(self, liked_ids, disliked_ids):
        payload = {"liked_ids": liked_ids, "disliked_ids": disliked_ids}
        try:
            response = requests.post(self.similarity_endpoint, json=payload)
            if response.status_code == 200:
                print("Data successfully sent to FastAPI")
                return response.json()
            else:
                print(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Failed to send data: {e}")

