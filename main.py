import lancedb
import pandas as pd
import pyarrow as pa


def lance_init():
    uri = "db/lancedb"
    db = lancedb.connect(uri)

    data = [
        {"vector": [1.1, 1.2], "lat": 45.5, "long": -122.7},
        {"vector": [0.2, 1.8], "lat": 40.1, "long": -74.1}
    ]

    db.create_table("my_table", data)

    print(db["my_table"].head().to_pandas())


lance_init()
