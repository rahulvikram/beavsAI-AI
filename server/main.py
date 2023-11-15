from typing import Union

from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    with open(f"../data/syllabus/{file.filename}", "wb") as local_file:
        local_file.write(file.file.read())
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "location": f"../data/syllabus/{file.filename}",
    }