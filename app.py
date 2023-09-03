from flask import Flask, render_template, request, make_response
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
import base64

app = Flask(__name__)

# Route to render the HTML form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route to handle the image upload and processing
@app.route('/upload', methods=['POST'])
def upload_image():

    class_names=['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy']
    try:
        if 'image' not in request.files:
            raise Exception('No file part')

        file = request.files['image']

        if file.filename == '':
            raise Exception('No selected file')
        
        model = load_model("C:\\Users\\ADMIN\\Downloads\\model.h5")

        # Load the uploaded image using Pillow
        img = Image.open(file)

        img = img.resize((120, 120))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        probs = model.predict(img)[0]

        # Get the predicted class index and name
        pred_class_prob = np.argmax(probs)
        pred_class_name = class_names[pred_class_prob]

        max_prob = np.max(probs)
        print(f'Predicted class: {pred_class_name}')
        print(f'Maximum probability: {max_prob}')

        # Display the image with the predicted class and probability
        plt.imshow(img[0])
        plt.axis('off')
        plt.text(5, 15, f'Predicted class: {pred_class_name}\nMaximum probability: {max_prob:.2f}', fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.8))
        plt.show()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Send the image as a response with the appropriate content type
        return Response(buffer.read(), content_type='image/png')

    except Exception as e:
        # Create a detailed error message with the exception description
        error_message = f'Error processing image: {str(e)}'

        # Create a response with a 500 Internal Server Error status code
        response = make_response(error_message, 500)

        # Set the response content type to text/plain
        response.headers['Content-Type'] = 'text/plain'

        return response

if __name__ == '__main__':
    app.run(debug=True)
