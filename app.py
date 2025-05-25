from flask import Flask, render_template, url_for

app = Flask(__name__)

IMAGE_BASE = "https://storage.googleapis.com/bucket-main-ta/static/images/image_{}.jpg"
VIDEO_BASE = "https://storage.googleapis.com/bucket-main-ta/static/videos/video_{}.mp4"

# Group definitions
IMAGE_GROUPS = {
    "trending": (0, 9),
    "popular": (10, 29),
    "general": (30, 79),
    "rare": (80, 99),
}

VIDEO_GROUPS = {
    "short-clips": (0, 9),
    "documentaries": (10, 29),
    "tutorials": (30, 59),
    "archived": (60, 65),
}

def make_urls(base_url, start, end):
    return [base_url.format(i) for i in range(start, end + 1)]

@app.route("/")
def home():
    return render_template('home.html', 
                         image_groups=IMAGE_GROUPS, 
                         video_groups=VIDEO_GROUPS)

@app.route("/images/<group>")
def show_images(group):
    if group not in IMAGE_GROUPS:
        return "Invalid image group", 404
    start, end = IMAGE_GROUPS[group]
    urls = make_urls(IMAGE_BASE, start, end)
    return render_template('images.html', group=group, urls=urls)

@app.route("/videos/<group>")
def show_videos(group):
    if group not in VIDEO_GROUPS:
        return "Invalid video group", 404
    start, end = VIDEO_GROUPS[group]
    urls = make_urls(VIDEO_BASE, start, end)
    return render_template('videos.html', group=group, urls=urls)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)