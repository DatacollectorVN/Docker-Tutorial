import pymysql
from app import app
from db import mysql
from flask import jsonify

@app.route('/')
def users():
    conn = mysql.connect()
    
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM user")
    
    rows = cursor.fetchall()
    
    resp = jsonify(rows)
    resp.status_code = 200
    
    return resp

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')