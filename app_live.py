from flask import Flask, request, jsonify
from utils.model_utils import load_model, save_model
from utils.data_utils import append_data, load_data
from utils.trainer import train_model, get_metrics
from apscheduler.schedulers.background import BackgroundScheduler
import os, time

app = Flask(__name__)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

model = load_model()

# Track last training timestamp
last_train_time = 0
TRAIN_INTERVAL = 60  # seconds

def auto_train():
    global model, last_train_time
    dataset = load_data()
    if not dataset:
        print("[NeoMind] No new data to train.")
        return
    print("[NeoMind] Auto-training started...")
    logs = train_model(model, dataset)
    save_model(model)
    last_train_time = time.time()
    print(f"[NeoMind] Auto-training finished: {logs[-1]}")

# Setup scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=auto_train, trigger="interval", seconds=TRAIN_INTERVAL)
scheduler.start()

@app.route('/')
def home():
    return jsonify({"NeoMind": "Online", "status": "ready", "version": "v1.1 live"})

@app.route('/upload-data', methods=['POST'])
def upload_data_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    append_data(data)
    return jsonify({"status": "Data added", "entries": len(data)})

@app.route('/train', methods=['POST'])
def manual_train():
    auto_train()
    return jsonify({"status": "Manual training triggered"})

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(get_metrics())

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "model_loaded": True,
        "training_ready": True,
        "last_training": last_train_time
    })

if __name__ == '__main__':
    print("[NeoMind] Starting Live Learning Server...")
    app.run(host='0.0.0.0', port=8000, debug=True)
