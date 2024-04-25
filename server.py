import lancedb
import pyarrow as pa
from flask import Flask, jsonify
from transformers import TFAutoModel, AutoTokenizer

app = Flask(__name__)

# 모델과 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
model = TFAutoModel.from_pretrained("klue/roberta-base")

# lanceDB init
uri = "db/lancedb"
db = lancedb.connect(uri)
schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), list_size=2))])
tbl = db.create_table("vector", schema=schema)


def insert_embedding(vector, item):
    data = [
        {"vector": vector, "item": item}
    ]
    tbl.add(data)


def embedding(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    print(embeddings)
    return embeddings.numpy().tolist()


@app.route('/')
def index():
    return "Welcome to the Flask App!"


@app.route('/item/<item_name>', methods=['POST'])
def item_embedding(item_name):
    embeddings = embedding(item_name)
    insert_embedding(embeddings, item_name)
    return jsonify({"item_name": item_name, "embedding": embeddings}), 200


if __name__ == '__main__':
    app.run(debug=True)
