from flask import Flask, request, jsonify
from PIL import Image
import torch
from io import BytesIO
import base64

app = Flask(__name__)

# Decide whether to use torch or tensorflow
model = torch.load("does_not_exist.pt")
model.eval()

@app.route("/generate", methods=["POST"])
def generate_image():
    file = request.data
    sketch_image = Image.open(BytesIO(file)).convert("L")  # Convert to grayscale

    # i dont have the model yet... awkward
    # preprocessed_image = preprocess(sketch_image)  # do i need this????
    # generated_image = model(preprocessed_image)

    # placeholder for now, just until frontend part is over
    generated_image = Image.new("RGB", (256, 256), color=(73, 109, 137))

    # base64 again
    buffered = BytesIO()
    generated_image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return encoded_img

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
