import json
import os
import pika
import shortuuid
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from flask import Flask, abort, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

# CONFIG
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PW = os.getenv('POSTGRES_PW', 'password')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'mydb')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_URL = f'postgresql://{POSTGRES_USER}:{POSTGRES_PW}@{POSTGRES_HOST}/{POSTGRES_DB}'

def create_database_if_not_exists():
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            user=POSTGRES_USER,
            password=POSTGRES_PW,
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (POSTGRES_DB,))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE "{POSTGRES_DB}"')
            print(f"Database '{POSTGRES_DB}' created successfully")
        else:
            print(f"Database '{POSTGRES_DB}' already exists")
            
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"Error creating database: {e}")
        raise
create_database_if_not_exists()

# INIT - This is crucial!
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['SQLALCHEMY_DATABASE_URI'] = POSTGRES_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

def get_rabbitmq_connection():
    url = os.environ.get('RABBITMQ_URL', 'amqp://guest:guest@rabbitmq:5672/')
    params = pika.URLParameters(url)
    return pika.BlockingConnection(params)

# MODELS
class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(24), unique=True, nullable=False)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    sessionId = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    input = db.Column(db.JSON, nullable=False)
    output = db.Column(db.JSON)
    predicted = db.Column(db.Boolean, default=False)

# CREATE TABLES IF THEY DON'T EXIST
def create_tables():
    with app.app_context():
        db.create_all()

# Initialize tables when the module is imported
create_tables()

# ENDPOINTS
# POST /sessions
@app.route('/sessions', methods=['POST'])
def create_session():
    token = shortuuid.ShortUUID().random(length=12)
    session = Session(token=token)
    db.session.add(session)
    db.session.commit()
    return jsonify({'token': token})

# GET /sessions/<sessionToken>/predictions
@app.route('/sessions/<string:sessionToken>/predictions', methods=['GET'])
def get_predictions(sessionToken):
    session = Session.query.filter_by(token=sessionToken).first()
    if not session:
        abort(404, 'Session not found')
    
    predictions = Prediction.query.filter_by(sessionId=session.id).order_by(Prediction.id.desc()).all()
    return jsonify([
        {
            'id': p.id,
            'input': p.input,
            'output': p.output,
            'predicted': p.predicted
        } for p in predictions
    ])

# POST /predictions
@app.route('/predictions', methods=['POST'])
def create_predictions():
    data = request.get_json()
    token = data.get('token')
    inputs = data.get('inputs', [])
    
    session = Session.query.filter_by(token=token).first()
    if not session:
        abort(404, 'Session not found')
    
    predictions = []
    for inp in inputs:
        prediction = Prediction(sessionId=session.id, input=inp, predicted=False)
        db.session.add(prediction)
        predictions.append(prediction)
    
    db.session.commit()
    
    # Publish to RabbitMQ
    try:
        conn = get_rabbitmq_connection()
        channel = conn.channel()
        channel.queue_declare(queue='predict_queue', durable=True)
        channel.basic_publish(
            exchange='',
            routing_key='predict_queue',
            body=str(session.id),
            properties=pika.BasicProperties(delivery_mode=2))
        conn.close()
        print(f"Successfully queued session {session.id} for prediction")
    except Exception as e:
        print(f"RabbitMQ error: {e}")
        # Consider returning an error response or retrying
        return jsonify({'error': 'Failed to queue predictions'}), 500
    
    return jsonify({'inserted': len(predictions)})

if __name__ == '__main__':
    app.run(debug=True)
