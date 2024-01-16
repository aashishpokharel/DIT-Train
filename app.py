from fastapi import FastAPI
from predict import predict
from pydantic import BaseModel
import detectron2
# import cv2
from PIL import Image

app = FastAPI()


class Item(BaseModel):
    image_url:str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/get_results/")
async def get_results(item : Item):
    """
    API to get the recommendations for the query.
    It takes query as input and returns the recommendations.
    # """
    # X = Image.open(item.image_url)
    # X = np.array(X)
    # print(X)
    # print("output")
    with open('test_img.png', 'rb') as f:
        image = f.read()
    predictions = predict(X)
    
    return predictions


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", ports=[6000, 8000])