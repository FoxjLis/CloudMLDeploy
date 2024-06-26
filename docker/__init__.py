from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import json
import logging
import uuid

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)


def rectangles_intersect(box1, box2):
    return not (box1['x2'] < box2['x1'] or box1['x1'] > box2['x2'] or box1['y2'] < box2['y1'] or box1['y1'] > box2['y2'])


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = '/uploads'

model = YOLO('best_x.pt')

@app.route('/')
def home():
    return "Hello, Flask!"


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
            model_output = model(file_path)
            results_json_str = model_output[0].tojson()

            logging.debug(f"Results JSON: {results_json_str}")

            image = Image.open(file_path)
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()

            class_colors = {
                1: 'blue',
                2: 'green',
            }

            results = json.loads(results_json_str)
            filtered_results = []

            for result in results:
                intersect = False
                for filtered in filtered_results:
                    if rectangles_intersect(result['box'], filtered['box']):
                        intersect = True
                        if result['confidence'] > filtered['confidence']:
                            filtered_results[filtered_results.index(filtered)] = result
                        break
                if not intersect:
                    filtered_results.append(result)

            for result in filtered_results:
                name = result['name']
                confidence = result['confidence']
                box = result['box']
                class_id = result['class']

                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                color = class_colors.get(class_id, 'red')

                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, y1 - 20), f'{name} ({confidence:.2f})', fill=color, font=font)

            random_filename = f"{uuid.uuid4()}.jpg"
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], random_filename)
            image.save(result_image_path)

            return jsonify({"results": filtered_results, "image_path": random_filename}), 200
        except FileNotFoundError:
            logging.error("File not found")
            return jsonify({"error": "File not found"}), 404
        except Exception as e:
            logging.error(f"Exception: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unexpected error"}), 500


@app.route('/result/<string:image>', methods=['GET'])
def get_result_image(image):
    result_path = '/uploads/' + image
    try:
        return send_file(result_path, mimetype='image/jpg')
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/healthcheck')
def healthcheck():
    return 'OK'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
