from flask import Flask, request
import json
from datetime import datetime
import os

app = Flask(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/save', methods=['POST'])
def save_instruction():
    data = request.json
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'instruction_{timestamp}.json'
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Saved: {filepath}')
    print(json.dumps(data, indent=2))
    return {'status': 'saved', 'file': filepath}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)