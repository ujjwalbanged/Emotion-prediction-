from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi import HTTPException
from keras.models import load_model
import numpy as np
import pickle
import re




model_path = "Artifacts/BiGRU_Modle.keras"


tokenizer_path = "Artifacts/tokenizer.pkl"


max_sequence_length = 1000


emotions_lables = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


EMOTION_EMOJIES = {
    "sadness": "😢",
    "joy": "😂",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲"
}



def preprocesses_text(text : str)-> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



class textInput(BaseModel):
    text:str = Field(
        ...,
        min_length=1, 
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"Example" : "I feel so happy and excited "}
    )



class predictionresponse(BaseModel):
    text : str
    predicted_emotion : str
    confidence : float
    all_probabilities : dict[str, float]

class healthresponse(BaseModel):
    status : str
    model_loaded : bool


# modle loading and lifespam managemnet
dl_model ={
}


@asynccontextmanager
async def lifespam(app : FastAPI):
    print("loading the model and tokenizer....")
    dl_model['BiGRU'] = load_model(model_path)
    with open(tokenizer_path, 'rb') as file:
        dl_model['tokenizer']  = pickle.load(file)
    print("model are loaded sucessfully....")

    yield

    dl_model.clear()


app = FastAPI(
    lifespan= lifespam
)

#mount the static file to the FASTAPI app



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory="static"), name="static")


#api endpoints


#Server ui at home page 
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')


#Health check endpoints
@app.get("/health", response_model=healthresponse)
def healthcheck():
    return healthresponse(status="Server is running", model_loaded=bool(dl_model))

#prediction Emotions endpoints
@app.post('/predict', response_model=predictionresponse)
def predict_emotion(text_input : textInput):

    # Clean the input sentences
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    cleaned_text = preprocesses_text(text_input.text)

    
    tokernizer_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokernizer_text,
        maxlen = max_sequence_length,
        padding='post',
        truncating='post'

    )

    probabilites = BiGRU_model.predict(padded_sequence)[0]

    top_emotion_index  = int(np.argmax(probabilites))

    all_probabilites = {
        label: float(prob) for prob, label in zip( probabilites, emotions_lables)
    }

    return predictionresponse(
        text = text_input.text,
        predicted_emotion = emotions_lables[top_emotion_index],
        confidence = float(probabilites[top_emotion_index]),
        all_probabilities = all_probabilites
    )