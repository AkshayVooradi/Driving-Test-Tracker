from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import os
import shutil
import mimetypes
from werkzeug.utils import secure_filename
from video_processing import process_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['PROCESSED_FOLDER']):
    os.makedirs(app.config['PROCESSED_FOLDER'])

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        print(f"Original file saved to: {file_path}")

        processed_file_path = process_video(file_path, app.config['PROCESSED_FOLDER'])

        print(f"Processed file path: {processed_file_path}")

        return render_template('upload.html', filename=os.path.basename(processed_file_path))
        # return render_template('upload.html', filename= "processed/processed_lane4_online-video-cutter.com.mp4")



@app.route('/processed/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(file_path):
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if unknown
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename, mimetype=mime_type)
    else:
        print(f"File not found: {file_path}")
        return "File not found", 404

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
