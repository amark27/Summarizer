from flask import Flask, request

app = Flask(__name__)

@app.route('/api/v1/', methods=['GET'])
def home():
    return f'Running on {request.host}...'

if __name__=='__main__':
    app.run(debug=True)