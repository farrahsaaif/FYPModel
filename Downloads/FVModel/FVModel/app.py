from flask import Flask, request, jsonify, send_file
import io
from PIL import Image
import cv2
from main import process_images  # Import your logic from main.py

app = Flask(__name__)

@app.route('/stylize', methods=['POST'])
def stylize():
    try:
        # Get images from the request
        content_image = request.files['content_image']
        design_image = request.files['design_image']

        # Convert to PIL images
        content_image = Image.open(content_image).convert("RGB")
        design_image = Image.open(design_image).convert("RGB")

        # Process the images
        output_image = process_images(
            model_path="mask_rcnn_IS_50.pth",  # Replace with your model's actual path
            content_image=content_image,
            design_patch=design_image
        )

        if output_image is None:
            return jsonify({"error": "No clothing items detected."}), 400

        # Convert OpenCV output to bytes for response
        _, buffer = cv2.imencode('.jpg', output_image)
        io_buffer = io.BytesIO(buffer)
        io_buffer.seek(0)

        return send_file(io_buffer, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
