from flask import Flask, Response, request
from flask_cors import CORS
from torch.cuda import device
from waitress import serve

import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)
cors = CORS(app) # Correction temporaire pour le javascript côté client
data = None

@app.route('/')
def is_up():
    return Response(None, 200)

@app.route('/request')
def make_request():
    req = request.args.get('prompt')
    if req is not None and len(req) > 0:
        return get_response(req)
    return Response(None, 400)


def get_data():
    global data
    if data is None:
        data = torch.load("data.pth")
    return data

def get_device() -> device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_modal() -> NeuralNet:
    data = get_data()

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(get_device())
    model.load_state_dict(model_state)
    model.eval()
    return model

def get_intents():
    with open('intents.json', 'r', encoding='UTF-8') as json_data:
        return json.load(json_data)

def get_response(msg) -> Response:
    data = get_data()
    all_words = data['all_words']
    tags = data['tags']
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(get_device())

    output = get_modal()(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in get_intents()['intents']:
            if tag == intent["tag"]:
                return Response(random.choice(intent['responses']), 200)

    return Response(None, 404)


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=9000)