import os
import aiohttp
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
print('started')
# Load CSV
csv_file = 'data\Products-Export-2025-February-16-1830-full.csv'
df = pd.read_csv(csv_file).head(1000)

# Ensure save directory exists
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)


async def download_image(session, img_id, url):
    filename = f"{img_id}.jpg"
    path = os.path.join(save_dir, filename)

    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(path, "wb") as f:
                    f.write(await response.read())
                return filename  # Return filename if successful
    except Exception as e:
        print(f"Failed: {url} -> {e}")
    return None  # Return None if failed


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, img_id, url) for img_id, url in zip(df["id"], df["url"])]
        return await tqdm_asyncio.gather(*tasks)


df["filename"] = asyncio.run(main())  # Store downloaded filenames in CSV

# Save updated CSV
df.to_csv("updated_images.csv", index=False)