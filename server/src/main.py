from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from playground import whisper_debug


app = FastAPI()

def main():
    print("This is the main method!")