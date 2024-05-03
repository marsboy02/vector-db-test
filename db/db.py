import psycopg2
from flask import jsonify, app
from psycopg2.extras import RealDictCursor


def get_db_connection():
    conn = psycopg2.connect(
        host='localhost',
        database='vector',
        user='vector',
        password='vector'
    )
    return conn


# @app.route('/data')
def get_data():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute('SELECT * FROM your_table_name LIMIT 5;')
    results = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(results)
