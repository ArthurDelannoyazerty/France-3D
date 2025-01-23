from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_coordinates')
def handle_coordinates(data):
    print(f"Received rectangle coordinates: {data}")

if __name__ == '__main__':
    socketio.run(app, debug=True)
