from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
from inference import ImageToWordModel
from mltu.configs import BaseModelConfigs
from gtts import gTTS
from textblob import TextBlob
import os
import base64

app = Flask(__name__)

# Load the model and any other necessary configurations
configs = BaseModelConfigs.load(r"C:\Users\bhara\Untitled Folder 6\configs.yaml")
image_to_word_model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Perform any additional preprocessing if needed
    return image

def generate_audio(text):
    tts = gTTS(text=text, lang='en', slow=False)
    audio_path = r"C:\Users\bhara\New folder\static\generated_audio.mp3"
    tts.save(audio_path)

    # Read audio file as binary and encode it as base64
    with open(audio_path, "rb") as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

    return audio_path, audio_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Check if the image is present in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"})

        file = request.files['image']

        # Check if the file has a filename
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        # Save the image to a temporary file
        temp_path = "temp_image.png"
        file.save(temp_path)

        # Preprocess the image
        image = preprocess_image(temp_path)

        # Perform prediction using the model
        prediction_text = image_to_word_model.predict(image)

        # Generate audio and get the audio file path and data
        prediction_text=str((TextBlob(prediction_text)).correct())
        audio_path, audio_data = generate_audio(prediction_text)

        # Return the prediction, audio file path, and audio data as JSON
        result = {"prediction": prediction_text, "audio_path": audio_path, "audio_data": audio_data}
        return jsonify(result)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return jsonify({"error": error_message})

@app.route('/play_audio')
def play_audio():
    audio_path = request.args.get('audio_path')
    return send_from_directory(os.path.dirname(audio_path), os.path.basename(audio_path), mimetype='audio/mp3')

if __name__ == "__main__":
    app.run(debug=True)
