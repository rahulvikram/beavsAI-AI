# © 2023 App Development Club @ Oregon State Unviersity
# All Rights Reserved

# This file exposes a REST API that can be used to interact with the Pinecone vector store. 
# It is intended to be used as a reference for how to use Pinecone in a production environment. 
# It is not intended to be used as a production server. 
# For example, it does not have any authentication or rate limiting.

# All of the functionality in this file can be used for free with the starter indexes/environment.

# However, if you're wanting to extend this and add expanded functionality, you
# will need to upgrade to a paid plan and put that in your .env file. 

import json
import logging
import os
import re
from typing import Union

import pinecone
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from constants import course_map, data_map

origins = [
    "http://localhost:8080",
    "http://localhost:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://0.0.0.0:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Our Server


class LoadDocuemnts(BaseModel):
    class_name: str


class IndexInfo(BaseModel):
    vectorCount: int
    dimension: int
    indexFullness: float
    totalVectorCount: int


class PineconeInfo(BaseModel):
    namespaces: list
    dimension: int
    indexFullness: float
    totalVectorCount: int


class GetResponse(BaseModel):
    class_name: str
    query: str


# Generate mapping for data
def generate_mapping(filenames):
    mapping = []
    for filename in filenames:
        crn_matcher = re.search(r"CS(\d+)_", filename)
        crn = crn_matcher.group(1) if crn_matcher else None
        term = (
            "F23"
            if filename.endswith("F23.pdf")
            else "S22"
            if filename.endswith("S22.pdf")
            else "unknown"
        )
        index = f"cs{crn.lower()}-index" if crn else None
        course_name = course_map[crn] if crn else None
        mapping.append(
            {
                "index": index,
                "filename": filename,
                "term": term,
                "crn": crn,
                "course_name": course_name,
            }
        )
    return mapping


# Add pdf to pinecone index
async def add_pdf_to_pinecone_index(pdf_path, index_name, embeddings):
    print("tmertn", index_name)
    if index_name is None:
        return {"error": "Index name is required"}

    existing_indexes = pinecone.list_indexes()

    if existing_indexes:
        if existing_indexes[0] != index_name:
            if existing_indexes[0] is not None:
                print("deleting index")
                print(existing_indexes[0])
                pinecone.delete_index(existing_indexes[0])
            pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
        else:
            print("index already exists")
            res = await upload_pdf(pdf_path, index_name, embeddings)
            print(res)
            return {"error": res["error"]}
    else:
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

    if index_name in pinecone.list_indexes():
        await upload_pdf(pdf_path, index_name, embeddings)

    return {"error": "PDF uploaded successfully"}


async def upload_pdf(pdf_path, index_name, embeddings):
    existing_indexes = pinecone.list_indexes()
    if existing_indexes:
        index_info = pinecone.Index(index_name).describe_index_stats()
        print(index_info)
        if index_info.namespaces == {}:
            vector_count = index_info.total_vector_count
            if vector_count == 0:
                loader = UnstructuredPDFLoader(pdf_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
                )
                texts = text_splitter.split_documents(data)
                docsearch = Pinecone.from_documents(
                    texts, embeddings, index_name=index_name
                )
                return {"error": "PDF uploaded successfully"}
            else:
                return {"error": "Index already has a document"}
        else:
            return {"error": "Index already has documents"}
    else:
        return {"error": "Index does not exist"}


# Load .env
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = os.environ["PINECONE_API_ENV"]

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV,
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# hello world
@app.get("/")
def read_root():
    return {"Hello": "World"}


# Upload syllabus
@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    with open(f"../data/syllabus/{file.filename}", "wb") as local_file:
        local_file.write(file.file.read())
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "location": f"../data/syllabus/{file.filename}",
    }


# Load documents into Pinecone
@app.post("/load_documents")
async def load_documents(load_documents: LoadDocuemnts):
    crn = None
    for item in data_map:
        if item["course_name"] == load_documents.class_name:
            crn = item["crn"]
            break
    syllabus_directory = "../data/syllabus"
    index_name = f"cs{crn.lower()}-index"
    pdf_path = None
    for item in data_map:
        if item["index"] == index_name:
            pdf_path = f"{syllabus_directory}/{item['filename']}"
            break
    status = await add_pdf_to_pinecone_index(pdf_path, index_name, embeddings)
    return {"status": status}


# Get q&a for a class classname w/ query
@app.post("/response")
async def get_response(get_response: GetResponse):
    index_name = f"cs{get_response.class_name.lower()}-index"
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    docs = docsearch.similarity_search(get_response.query)
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    llm_response = chain.run(input_documents=docs, question=get_response.query)
    return {"response": llm_response}


# List all indexes
@app.get("/list_indexes")
async def list_indexes():
    return pinecone.list_indexes()


# Get information about an index
@app.get("/list_index_info/{index_name}")
async def list_index_info(index_name: str):
    return pinecone.describe_index(index_name)


# Get all available data
@app.get("/available_data")
def available_data():
    return data_map


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8080, log_level="info", reload=True)
