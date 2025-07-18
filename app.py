from flask import Flask, render_template, url_for, request, make_response, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import hashlib
import threading
import time
import requests
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/feb/Documents/GitHub/mock-flask-app/gcs-key.json"
from werkzeug.utils import secure_filename
from google.cloud import storage

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'supersecretkey'  # Needed for flash messages
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = SQLAlchemy(app)

class UserContent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    url = db.Column(db.String(300), nullable=False)
    content_type = db.Column(db.String(10), nullable=False)  # 'image' or 'video'
    group = db.Column(db.String(50), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<UserContent {self.filename}>'

# CDN configuration based on user location
CDN_ENDPOINT = "https://cdn.sohryuu.me"

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

CDN_HEALTHCHECK_URL = f"{CDN_ENDPOINT}/static/images/image_0.jpg"  # Use a small, always-present file
CDN_HEALTHCHECK_INTERVAL = 30  # seconds
CDN_AVAILABLE = True  # Global flag


def cdn_healthcheck_loop():
    global CDN_AVAILABLE
    while True:
        try:
            resp = requests.head(CDN_HEALTHCHECK_URL, timeout=3)
            CDN_AVAILABLE = resp.status_code == 200
        except Exception:
            CDN_AVAILABLE = False
        time.sleep(CDN_HEALTHCHECK_INTERVAL)

# Start healthcheck thread at startup
def start_cdn_healthcheck():
    t = threading.Thread(target=cdn_healthcheck_loop, daemon=True)
    t.start()

def make_urls(base_url, start, end, region='default', cdn_active=None):
    """Generate URLs - use CDN if active, else fallback to original storage URL"""
    if cdn_active is None:
        cdn_active = CDN_AVAILABLE
    if cdn_active:
        cdn_url = base_url.replace('https://storage.googleapis.com/bucket-main-ta', CDN_ENDPOINT)
    else:
        cdn_url = base_url  # Use original storage URL
    return [cdn_url.format(i) for i in range(start, end + 1)]

def add_cache_headers(response, cache_type='static'):
    """Enhanced cache headers with better control"""
    if cache_type == 'static':
        response.headers['Cache-Control'] = 'public, max-age=2592000, immutable'  # 30 days
        response.headers['Expires'] = (datetime.utcnow() + timedelta(days=30)).strftime('%a, %d %b %Y %H:%M:%S GMT')
    elif cache_type == 'media':
        response.headers['Cache-Control'] = 'public, max-age=2592000'  # 30 days
        response.headers['Expires'] = (datetime.utcnow() + timedelta(days=30)).strftime('%a, %d %b %Y %H:%M:%S GMT')
    elif cache_type == 'page':
        response.headers['Cache-Control'] = 'public, max-age=3600, stale-while-revalidate=300'  # 1 hour
    elif cache_type == 'api':
        response.headers['Cache-Control'] = 'public, max-age=300'  # 5 minutes
    
    # Enhanced ETag
    content_hash = hashlib.md5(response.get_data()).hexdigest()[:16]
    response.headers['ETag'] = f'W/"{content_hash}"'
    
    # Add security and optimization headers
    response.headers['Vary'] = 'Accept-Encoding, X-User-Region, Accept'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    return response

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file_to_gcs(file_stream, filename, content_type, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f'static/uploads/{filename}')
    blob.upload_from_file(file_stream, content_type=content_type)
    # Do NOT call blob.make_public() if uniform bucket-level access is enabled
    return f'https://storage.googleapis.com/{bucket_name}/static/uploads/{filename}'

@app.route('/upload', methods=['GET', 'POST'])
def upload_content():
    if request.method == 'POST':
        file = request.files.get('file')
        content_type = request.form.get('content_type')
        group = request.form.get('group')
        if not file or file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('File type not allowed!', 'danger')
            return redirect(request.url)
        if content_type not in ['image', 'video']:
            flash('Invalid content type!', 'danger')
            return redirect(request.url)
        if not group:
            flash('Group is required!', 'danger')
            return redirect(request.url)
        filename = secure_filename(str(file.filename))
        # Upload to GCS
        public_url = f"https://storage.googleapis.com/bucket-main-ta/static/uploads/{filename}"
        new_content = UserContent(filename=filename, url=public_url, content_type=content_type, group=group)
        db.session.add(new_content)
        db.session.commit()
        flash('Upload successful!', 'success')
        if content_type == 'image':
            return redirect(url_for('show_images', group=group))
        else:
            return redirect(url_for('show_videos', group=group))
    # GET: show upload form
    return render_template('upload.html', image_groups=IMAGE_GROUPS, video_groups=VIDEO_GROUPS)

@app.route("/")
def home():
    response = make_response(render_template('home.html', 
                                           image_groups=IMAGE_GROUPS, 
                                           video_groups=VIDEO_GROUPS))
    return add_cache_headers(response, 'page')

@app.route("/images/<group>")
def show_images(group):
    if group not in IMAGE_GROUPS:
        return "Invalid image group", 404
    cdn_region = request.headers.get('X-CDN-Region', 'Unknown')
    if 'EU' in cdn_region:
        cdn_flag = '🇪🇺'
    elif 'Asia' in cdn_region:
        cdn_flag = '🇮🇩'
    elif 'US' in cdn_region:
        cdn_flag = '🇺🇸'
    else:
        cdn_flag = '🌐'
    start, end = IMAGE_GROUPS[group]
    IMAGE_BASE = "https://storage.googleapis.com/bucket-main-ta/static/images/image_{}.jpg"
    urls = make_urls(IMAGE_BASE, start, end)
    user_images = UserContent.query.filter_by(content_type='image', group=group).order_by(UserContent.uploaded_at.desc()).all()
    response = make_response(render_template('images.html', group=group, urls=urls, CDN_REGION=cdn_region, CDN_FLAG=cdn_flag, CDN_ACTIVE=CDN_AVAILABLE, user_images=user_images))
    return add_cache_headers(response, 'page')

@app.route("/videos/<group>")
def show_videos(group):
    if group not in VIDEO_GROUPS:
        return "Invalid video group", 404
    cdn_region = request.headers.get('X-CDN-Region', 'Unknown')
    if 'EU' in cdn_region:
        cdn_flag = '🇪🇺'
    elif 'Asia' in cdn_region:
        cdn_flag = '🇮🇩'
    elif 'US' in cdn_region:
        cdn_flag = '🇺🇸'
    else:
        cdn_flag = '🌐'
    start, end = VIDEO_GROUPS[group]
    VIDEO_BASE = "https://storage.googleapis.com/bucket-main-ta/static/videos/video_{}.mp4"
    urls = make_urls(VIDEO_BASE, start, end)
    user_videos = UserContent.query.filter_by(content_type='video', group=group).order_by(UserContent.uploaded_at.desc()).all()
    response = make_response(render_template('videos.html', group=group, urls=urls, CDN_REGION=cdn_region, CDN_FLAG=cdn_flag, CDN_ACTIVE=CDN_AVAILABLE, user_videos=user_videos))
    return add_cache_headers(response, 'page')

@app.route('/edit/<int:content_id>', methods=['GET', 'POST'])
def edit_content(content_id):
    content = UserContent.query.get_or_404(content_id)
    if request.method == 'POST':
        group = request.form.get('group')
        new_filename = request.form.get('filename')
        file = request.files.get('file')
        if not group:
            flash('Group is required!', 'danger')
            return redirect(request.url)
        if not new_filename:
            flash('Filename is required!', 'danger')
            return redirect(request.url)
        new_filename = secure_filename(str(new_filename))
        # Handle file replacement
        if file and file.filename:
            if not allowed_file(file.filename):
                flash('File type not allowed!', 'danger')
                return redirect(request.url)
            # Upload new file to GCS
            public_url = upload_file_to_gcs(file.stream, new_filename, file.content_type, 'bucket-main-ta')
            content.filename = new_filename
            content.url = public_url
        else:
            # Only update filename in GCS if changed (rename not supported, so re-upload is needed for true rename)
            if new_filename != content.filename:
                flash('To change filename, please re-upload the file with the new name.', 'warning')
                return redirect(request.url)
        content.group = group
        db.session.commit()
        flash('Content updated!', 'success')
        if content.content_type == 'image':
            return redirect(url_for('show_images', group=content.group))
        else:
            return redirect(url_for('show_videos', group=content.group))
    return render_template('edit_content.html', content=content, image_groups=IMAGE_GROUPS, video_groups=VIDEO_GROUPS)

@app.route('/delete/<int:content_id>', methods=['POST'])
def delete_content(content_id):
    content = UserContent.query.get_or_404(content_id)
    group = content.group
    content_type = content.content_type
    # Remove file from disk
    try:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], content.filename))
    except Exception:
        pass
    db.session.delete(content)
    db.session.commit()
    flash('Content deleted!', 'success')
    if content_type == 'image':
        return redirect(url_for('show_images', group=group))
    else:
        return redirect(url_for('show_videos', group=group))

# Health check endpoint for CDN
@app.route("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Error handlers
@app.errorhandler(404)
def not_found(error):
    response = make_response(render_template('404.html'), 404)
    return add_cache_headers(response, 'page')

@app.errorhandler(500)
def server_error(error):
    response = make_response(render_template('500.html'), 500)
    response.headers['Cache-Control'] = 'no-cache'
    return response

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    start_cdn_healthcheck()
    app.run(debug=True, host="0.0.0.0", port=8000)