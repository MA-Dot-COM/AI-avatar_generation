from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from yolo5.Avatar import User_Avatar, img_download

app = FastAPI()

class Item(BaseModel):
    name: Union[str, None] = None
    price: Union[float, None] = None
    url: str
    is_offer: Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put("/test")
def test_model(item: Item):
    url = item.url
    img_path = img_download(url)
    return User_Avatar(img_path,classification_model)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.price, "item_id": item_id}
