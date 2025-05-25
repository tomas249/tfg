"""
Worker service for processing neuron prediction requests.
Consumes messages from RabbitMQ and runs GNN model inference.
"""

import os
import json
import pika
import psycopg2
import pandas as pd
import time
from common.models import GNNModel, NeuronInferenceModel

# Configuration
MODEL_BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH", "common/neuron_model_bundle.pt")
RABBITMQ_URL = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@rabbitmq:5672/')
PG_CONN = {
   'host': os.getenv('POSTGRES_HOST', 'localhost'),
   'user': os.getenv('POSTGRES_USER', 'postgres'),
   'password': os.getenv('POSTGRES_PW', 'postgres'),
   'dbname': os.getenv('POSTGRES_DB', 'mydb')
}

# Load model once at startup
print("Loading model...")
model_bundle = NeuronInferenceModel.load(MODEL_BUNDLE_PATH)
print("Model loaded successfully")


def process_predictions(session_id):
   """Process all unpredicted rows for a given session."""
   conn = psycopg2.connect(**PG_CONN)
   cur = conn.cursor()
   
   try:
       # Get all unpredicted rows for this session
       cur.execute("""
           SELECT id, input FROM predictions
           WHERE "sessionId" = %s AND predicted = FALSE
       """, (session_id,))
       rows = cur.fetchall()
       
       if not rows:
           print(f"No unpredicted rows for session {session_id}")
           return
       
       # Extract IDs and inputs
       ids = []
       inputs = []
       for row_id, input_blob in rows:
           ids.append(row_id)
           inputs.append(input_blob)
       
       # Convert to DataFrame
       df = pd.DataFrame(inputs)
       if 'index' in df.columns:
           df.set_index('index', inplace=True)
       
       # Run model prediction
       print(f"Running predictions for {len(df)} inputs...")
       result = model_bundle.predict(df)
       result = result[[col for col in result.columns if col.startswith("pred_")]]
       result_records = result.to_dict(orient="records")
       
       # Update predictions table
       for pred_id, output_obj in zip(ids, result_records):
           cur.execute(
               "UPDATE predictions SET output = %s, predicted = TRUE WHERE id = %s",
               (json.dumps(output_obj), pred_id)
           )
       
       conn.commit()
       print(f"Updated {len(ids)} predictions for session {session_id}")
       
   except Exception as e:
       print(f"Error in process_predictions: {e}")
       conn.rollback()
       raise
   finally:
       conn.close()


def callback(ch, method, properties, body):
   """RabbitMQ message callback handler."""
   session_id = body.decode()
   print(f"Received session_id: {session_id}")
   
   try:
       process_predictions(session_id)
       print(f"Successfully processed session {session_id}")
   except Exception as e:
       print(f"Error processing predictions for session {session_id}: {e}")
   
   ch.basic_ack(delivery_tag=method.delivery_tag)


def connect_to_rabbitmq_with_retry(max_retries=5, delay=5):
   """Connect to RabbitMQ with retry logic."""
   for attempt in range(max_retries):
       try:
           print(f"Attempting to connect to RabbitMQ (attempt {attempt + 1}/{max_retries})")
           params = pika.URLParameters(RABBITMQ_URL)
           conn = pika.BlockingConnection(params)
           print("✓ Connected to RabbitMQ successfully")
           return conn
       except pika.exceptions.AMQPConnectionError as e:
           print(f"Failed to connect to RabbitMQ: {e}")
           if attempt < max_retries - 1:
               print(f"Retrying in {delay} seconds...")
               time.sleep(delay)
           else:
               print("Max retries reached. Exiting.")
               raise


def main():
   """Main worker execution loop."""
   print("Starting neuron prediction worker...")
   
   # Connect to RabbitMQ
   conn = connect_to_rabbitmq_with_retry()
   channel = conn.channel()
   
   # Declare queue
   channel.queue_declare(queue='predict_queue', durable=True)
   channel.basic_qos(prefetch_count=1)
   channel.basic_consume(queue='predict_queue', on_message_callback=callback)
   
   print(' [*] Waiting for messages. To exit press CTRL+C')
   
   try:
       channel.start_consuming()
   except KeyboardInterrupt:
       print("Stopping consumer...")
       channel.stop_consuming()
       conn.close()
       print("Worker stopped")


if __name__ == '__main__':
   main()
