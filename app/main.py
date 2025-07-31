from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.model import load_crf_model
from src.feature_extraction import prepare_single_sentence
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load your trained CRF model
crf = load_crf_model()

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, sentence: str = Form(...)):
    features = prepare_single_sentence(sentence)
    prediction = crf.predict([features])[0]
    words = sentence.split()
    result = list(zip(words, prediction))
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "sentence": sentence
    })

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
