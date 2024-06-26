from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'C:/Users/Karma/Desktop/test/gitt/backend/'

model = YOLO('best_x.pt')


@app.route('/whatsimplant', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            results = []
            results.append((model(file_path, save=True, save_txt = True)[0].tojson()))
            results.append(file_path)
            return jsonify({"results": results}), 200
        except FileNotFoundError:
            return jsonify({"error": "File not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unexpected error"}), 500


if __name__ == '__main__':
    app.run(debug=True)
