from flask import Flask, render_template, request, jsonify, send_file
import base64
import numpy as np
from PIL import Image, ImageOps
import io
import os
from models.sketch2img import generate_from_prompt, get_random_sample

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    canvas_data = data['canvas']
    prompt = data['prompt']

    canvas_image = Image.open(io.BytesIO(base64.b64decode(canvas_data.split(',')[1])))
    generated_image = generate_from_prompt(canvas_image, prompt)

    img_byte_arr = io.BytesIO()
    generated_image.save(img_byte_arr, format='PNG')
    encoded_image = f"data:image/png;base64,{base64.b64encode(img_byte_arr.getvalue()).decode()}"

    return jsonify({'generated_image': encoded_image})


@app.route('/random_sample', methods=['GET'])
def random_sample():
    random_drawing, generated_image = get_random_sample()
    def encode_image(img):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(img_byte_arr.getvalue()).decode()}"
    
    return jsonify({
        'random_drawing': encode_image(random_drawing),
        'generated_image': encode_image(generated_image)
    })

@app.route('/save', methods=['POST'])
def save_images():
    canvas_data = request.json['canvas']
    generated_data = request.json['generated']

    canvas_image = Image.open(io.BytesIO(base64.b64decode(canvas_data.split(',')[1])))
    generated_image = Image.open(io.BytesIO(base64.b64decode(generated_data.split(',')[1])))

    combined_width = canvas_image.width + generated_image.width
    combined_image = Image.new('RGB', (combined_width, canvas_image.height))
    combined_image.paste(canvas_image, (0, 0))
    combined_image.paste(generated_image, (canvas_image.width, 0))

    save_path = 'static/combined_image.png'
    combined_image.save(save_path)
    return send_file(save_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
