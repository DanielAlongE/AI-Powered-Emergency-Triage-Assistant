from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.routes import router
from .database import engine, Base
from . import models

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Triage Assistant Online POC", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

@app.get("/")
def read_root():
    return {"Hello": "World"}
