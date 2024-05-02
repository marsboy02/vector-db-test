import lancedb
import pyarrow as pa
from flask import Flask, jsonify
from transformers import TFAutoModel, AutoTokenizer

# 모델과 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
model = TFAutoModel.from_pretrained("klue/roberta-base")
print('model is ready!')

# lanceDB init
uri = "db/lancedb"
db = lancedb.connect(uri)

dim = 768
schema = pa.schema(
    [
        pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
        pa.field("item", pa.string())
    ]
)

tbl = db.create_table("vector", schema=schema, exist_ok=True)
tbl.create_fts_index("item", replace=True)
print("vector db is ready!")

app = Flask(__name__)
print('server is ready!')


def insert_embedding(vector, item):
    data = [{"vector": vector[0], "item": item}]
    tbl.add(data)


def embedding(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.numpy().tolist()


@app.route('/')
def index():
    return "Welcome to the Flask App!"


@app.route('/item/<item_name>', methods=['POST'])
def item_embedding(item_name):
    embeddings = embedding(item_name)
    insert_embedding(embeddings, item_name)
    return jsonify({"item_name": item_name, "embedding": embeddings}), 200


@app.route('/item', methods=['GET'])
def read_item_all():
    data = tbl.search([350000, 350000]).limit(dim).to_pandas()
    return data


@app.route('/item/<item_name>', methods=['GET'])
def search_text(item_name):
    return tbl.search(item_name).limit(10).select(["item"]).to_list()


if __name__ == '__main__':
    app.run(debug=False)
