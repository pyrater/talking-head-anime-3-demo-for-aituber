from flask import Flask, render_template
import time
import cv2
import numpy as np
import random
import time

app = Flask(__name__)


def generate_image_data():
    # Generate a simple image with a random FPS value overlaid
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    fps = random.randint(1, 1000)
    fps_text = f"FPS: {fps}"
    cv2.putText(image, fps_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the image to JPEG format
    _, jpeg_data = cv2.imencode('.jpg', image)
    return jpeg_data.tobytes()
    
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_feed')
def image_feed():
    def generate():
        while True:
            # Generate or fetch the image data
            # Replace the following line with your image generation code
            image_data = generate_image_data()
            

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image_data + b'\r\n')

            # Delay to achieve desired FPS
            time.sleep(0.2)  # 1 / 5 = 0.2

    return app.response_class(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
