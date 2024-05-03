import lancedb
import pyarrow as pa
from flask import Flask, jsonify
from transformers import TFAutoModel, AutoTokenizer

sample = [
        "완전 맛있는 햄버거",
        "패티가 맛있는 햄버거",
        "감자튀김과 함께 먹으면 맛있는 햄버거",
        "패스트푸드점에서 가장 유명한 햄버거",
        "검은색 콜라",
        "칼로리가 낮은 콜라",
        "치킨과 함께 먹으면 굉장히 맛있는 콜라",
        "패스트푸드와 먹기에는 별로인 콜라",
        "바삭바삭한 치킨",
        "축구를 보면서 먹으면 맛있는 치킨",
        "다이어트 하는 사람을 위한 구운 치킨",
        "콜라와 함께 먹기 좋은 양념 치킨",
        "달달한 고구마 피자",
        "느끼한 치즈 피자",
        "평범하게 먹기 좋은 콤비네이션 피자",
        "치킨과 먹으면 맛있는 파인애플 피자",
        "콜라와 함께 먹으면 맛있는 피자"
    ]

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


@app.route('/item/default', methods=['POST'])
def default_item_embedding():
    for item in sample:
        item_embedding(item)
    return jsonify({"message": "default data created"}), 200


@app.route('/item', methods=['GET'])
def read_item_all():
    return tbl.search([500000, 500000]).limit(dim).to_pandas()


@app.route('/item/<item_name>', methods=['GET'])
def search_text(item_name):
    return tbl.search(item_name).limit(10).select(["item"]).to_list()


if __name__ == '__main__':
    app.run(debug=False)
