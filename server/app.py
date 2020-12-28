from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import json

app = Flask(__name__)

@app.route('/api/v1/', methods=['GET'])
def home():
    return f'Running on {request.host}...'

@app.route('/api/v1/summarize/', methods=['POST'])
def summarize():
    return {'summary': 'test'}

@app.errorhandler(HTTPException)
def resource_not_found(err):
    return {'code': err.code, 
            'error': str(err)}, err.code

if __name__ == '__main__':
    app.run(debug=True)