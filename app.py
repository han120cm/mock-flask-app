from flask import Flask, send_from_directory, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
        <h1>Media Test Site (Origin Server)</h1>
        <h2>Images</h2>
        <img src="/static/images/image1.png" width="400"><br>
        <img src="/static/images/image2.png" width="400"><br>
        <h2>Videos</h2>
        <video width="400" controls>
          <source src="/static/videos/video1.mp4" type="video/mp4">
        </video><br>
        <video width="400" controls>
          <source src="/static/videos/video2.mp4" type="video/mp4">
        </video>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
