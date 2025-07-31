# app/main.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.model import load_model
from src.feature_extraction import prepare_single_sentence 
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load trained CRF model
crf = load_model()

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, sentence: str = Form(...)):
    X = prepare_single_sentence(sentence)
    pred = crf.predict([X])[0]
    words = sentence.split()
    result = list(zip(words, pred))
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "sentence": sentence
    })

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
