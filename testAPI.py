from typing import Optional
from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def read_root():
    return{"Hello": "World"}

@app.get("/user/{user_id}")
def user_item(user_id: int, q: Optional[str] = None):
    return {"user_id": user_id, "q": q}