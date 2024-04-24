from flask import Flask, jsonify
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)


# 데이터베이스 설정
def get_db_connection():
    conn = psycopg2.connect(
        host='localhost',
        database='vector',
        user='vector',
        password='vector'
    )
    return conn


# 메인 페이지 라우트
@app.route('/')
def index():
    return "Welcome to the Flask App!"


# 데이터베이스 연결 테스트 라우트
@app.route('/data')
def get_data():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute('SELECT * FROM your_table_name LIMIT 5;')
    results = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
