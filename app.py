from flask import Flask, render_template, url_for, request, make_response, redirect, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import hashlib
import threading
import time
import requests
import os
import logging
from dotenv import load_dotenv
from sqlalchemy import case
import pandas as pd
import subprocess

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use Cloud Run's default service account (no credentials file needed)
# The service account will be automatically authenticated

from werkzeug.utils import secure_filename

# Import Google Cloud Storage with error handling
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
    logger.info("Google Cloud Storage imported successfully")
except ImportError as e:
    logger.warning(f"Google Cloud Storage not available: {e}")
    GCS_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error importing Google Cloud Storage: {e}")
    GCS_AVAILABLE = False

app = Flask(__name__)

# [START cloud_sql_python_connector_postgres_pg8000]
# The following code is modified from the Google Cloud SQL Python Connector documentation
# and is licensed under the Apache 2.0 License.
# See: https://github.com/GoogleCloudPlatform/cloud-sql-python-connector
import pg8000
from google.cloud.sql.connector import Connector, IPTypes

from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize the Cloud SQL Python Connector
connector = Connector()

# Function to get the database connection
def getconn() -> "pg8000.dbapi.Connection":
    # Retrieve environment variables
    cloud_sql_connection_name = os.environ.get("CLOUD_SQL_CONNECTION_NAME")
    db_user = os.environ.get("DB_USER")
    db_pass = os.environ.get("DB_PASS")
    db_name = os.environ.get("DB_NAME")

    # Ensure all required environment variables are set
    if not all([cloud_sql_connection_name, db_user, db_pass, db_name]):
        raise ValueError("Missing required database environment variables.")

    conn: pg8000.dbapi.Connection = connector.connect(
        cloud_sql_connection_name, 
        "pg8000",
        user=db_user,
        password=db_pass,          
        db=db_name,                 
        ip_type=IPTypes.PUBLIC,                   # Use public IP for Cloud Run
    )
    return conn

# Configure the SQLAlchemy engine using the connection pool
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql+pg8000://"
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "creator": getconn,
}

db.init_app(app)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['GCS_BUCKET_NAME'] = "bucket-main-ta"

class UserContent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    url = db.Column(db.String(300), nullable=False)
    content_type = db.Column(db.String(10), nullable=False)  # 'image' or 'video'
    group = db.Column(db.String(50), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<UserContent {self.filename}>'

class PerformanceMetric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    latency = db.Column(db.Integer)  # in milliseconds
    region = db.Column(db.String(50))
    cache_status = db.Column(db.String(20))  # HIT, MISS, EXPIRED
    content_type = db.Column(db.String(20))  # image, video, page
    user_agent = db.Column(db.String(200))
    ip_address = db.Column(db.String(45))

    def __repr__(self):
        return f'<PerformanceMetric {self.region}:{self.latency}ms>'

with app.app_context():
    db.create_all()

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
            # logger.info(f"CDN healthcheck: {CDN_AVAILABLE}")
        except Exception as e:
            CDN_AVAILABLE = False
            logger.warning(f"CDN healthcheck failed: {e}")
        time.sleep(CDN_HEALTHCHECK_INTERVAL)

# Start healthcheck thread at startup
def start_cdn_healthcheck():
    try:
        t = threading.Thread(target=cdn_healthcheck_loop, daemon=True)
        t.start()
        # logger.info("CDN healthcheck thread started")
    except Exception as e:
        logger.error(f"Failed to start CDN healthcheck: {e}")

def make_urls(base_url, start, end, region='default', cdn_active=None):
    """Generate URLs - use CDN if active, else fallback to original storage URL"""
    try:
        if cdn_active is None:
            cdn_active = CDN_AVAILABLE
        if cdn_active:
            cdn_url = base_url.replace('https://storage.googleapis.com/bucket-main-ta', CDN_ENDPOINT)
        else:
            cdn_url = base_url  # Use original storage URL
        return [cdn_url.format(i) for i in range(start, end + 1)]
    except Exception as e:
        logger.error(f"Error generating URLs: {e}")
        return []

def add_cache_headers(response, cache_type='static'):
    """Enhanced cache headers with better control"""
    try:
        if cache_type == 'static':
            response.headers['Cache-Control'] = 'public, max-age=2592000, immutable'  # 30 days
            response.headers['Expires'] = (datetime.utcnow() + timedelta(days=30)).strftime('%a, %d %b %Y %H:%M:%S GMT')
        elif cache_type == 'media':
            response.headers['Cache-Control'] = 'public, max-age=2592000'  # 30 days
            response.headers['Expires'] = (datetime.utcnow() + timedelta(days=30)).strftime('%a, %d %b %Y %H:%M:%S GMT')
        elif cache_type == 'page':
            response.headers['Cache-Control'] = 'public, max-age=3060, stale-while-revalidate=10'
        elif cache_type == 'api':
            response.headers['Cache-Control'] = 'public, max-age=300'  # 5 minutes
        
        # Enhanced ETag
        content_hash = hashlib.md5(response.get_data()).hexdigest()[:16]
        response.headers['ETag'] = f'W/"{content_hash}"'
        
        # Add security and optimization headers
        response.headers['Vary'] = 'Accept-Encoding, X-User-Region, Accept'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        return response
    except Exception as e:
        logger.error(f"Error adding cache headers: {e}")
        return response

def replace_with_cdn(gcs_url):
    """Replaces a GCS URL with a CDN URL if the CDN is available."""
    if CDN_AVAILABLE:
        return gcs_url.replace('https://storage.googleapis.com/bucket-main-ta', CDN_ENDPOINT)
    return gcs_url

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file_to_gcs(file_stream, filename, content_type, bucket_name):
    if not GCS_AVAILABLE:
        logger.error("Google Cloud Storage is not available. Cannot upload file.")
        raise Exception("Google Cloud Storage is not available.")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f'static/uploads/{filename}')
        blob.upload_from_file(file_stream, content_type=content_type)
        # Do NOT call blob.make_public() if uniform bucket-level access is enabled
        return f'https://storage.googleapis.com/{bucket_name}/static/uploads/{filename}'
    except Exception as e:
        logger.error(f"Error uploading file to GCS: {e}")
        raise

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
        gcs_url = upload_file_to_gcs(file.stream, filename, file.content_type, app.config['GCS_BUCKET_NAME'])
        cdn_url = replace_with_cdn(gcs_url)
        new_content = UserContent(filename=filename, url=cdn_url, content_type=content_type, group=group)  # type: ignore
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
    try:
        response = make_response(render_template('home.html', 
                                               image_groups=IMAGE_GROUPS, 
                                               video_groups=VIDEO_GROUPS))
        return add_cache_headers(response, 'page')
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        return "Internal server error", 500

@app.route("/images/<group>")
def show_images(group):
    try:
        if group not in IMAGE_GROUPS:
            logger.warning(f"Invalid image group requested: {group}")
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
        
        # Safely query the database
        try:
            user_images = UserContent.query.filter_by(content_type='image', group=group).order_by(UserContent.uploaded_at.desc()).all()
        except Exception as e:
            logger.error(f"Database error in show_images: {e}")
            user_images = []
        
        response = make_response(render_template('images.html', 
                                               group=group, 
                                               urls=urls, 
                                               CDN_REGION=cdn_region, 
                                               CDN_FLAG=cdn_flag, 
                                               CDN_ACTIVE=CDN_AVAILABLE, 
                                               user_images=user_images))
        return add_cache_headers(response, 'page')
    except Exception as e:
        logger.error(f"Error in show_images route: {e}")
        return "Internal server error", 500

@app.route("/videos/<group>")
def show_videos(group):
    try:
        if group not in VIDEO_GROUPS:
            logger.warning(f"Invalid video group requested: {group}")
            return "Invalid video group", 404
        
        cdn_region = request.headers.get('X-CDN-Region', 'Origin')
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
        
        # Safely query the database
        try:
            user_videos = UserContent.query.filter_by(content_type='video', group=group).order_by(UserContent.uploaded_at.desc()).all()
        except Exception as e:
            logger.error(f"Database error in show_videos: {e}")
            user_videos = []
        
        response = make_response(render_template('videos.html', 
                                               group=group, 
                                               urls=urls, 
                                               CDN_REGION=cdn_region, 
                                               CDN_FLAG=cdn_flag, 
                                               CDN_ACTIVE=CDN_AVAILABLE, 
                                               user_videos=user_videos))
        return add_cache_headers(response, 'page')
    except Exception as e:
        logger.error(f"Error in show_videos route: {e}")
        return "Internal server error", 500

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

        # Update filename if it has changed
        if new_filename != content.filename:
            content.filename = new_filename

        # Handle file replacement
        if file and file.filename:
            if not allowed_file(file.filename):
                flash('File type not allowed!', 'danger')
                return redirect(request.url)
            # Upload new file to GCS
            gcs_url = upload_file_to_gcs(file.stream, new_filename, file.content_type, app.config['GCS_BUCKET_NAME'])
            cdn_url = replace_with_cdn(gcs_url)
            content.url = cdn_url
        
        # Add a cache-busting parameter to the URL to force CDN/browser refresh
        content.url = f"{content.url}?v={int(time.time())}"

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
    
    # Correctly delete the file from Google Cloud Storage
    if GCS_AVAILABLE:
        try:
            client = storage.Client()
            bucket = client.bucket(app.config["GCS_BUCKET_NAME"])  # Use config variable
            blob = bucket.blob(f'static/uploads/{content.filename}')
            blob.delete()
            flash(f'File {content.filename} deleted from GCS.', 'info')
        except Exception as e:
            logger.error(f"Failed to delete {content.filename} from GCS: {e}")
            flash('Error deleting file from cloud storage.', 'danger')

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
    try:
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {"status": "unhealthy", "error": str(e)}, 500

# Debug endpoint to help identify issues
@app.route("/debug")
def debug_info():
    try:
        debug_info = {
            "database_path": app.config['SQLALCHEMY_DATABASE_URI'],
            "upload_folder": app.config['UPLOAD_FOLDER'],
            "gcs_available": GCS_AVAILABLE,
            "cdn_available": CDN_AVAILABLE,
            "image_groups": list(IMAGE_GROUPS.keys()),
            "video_groups": list(VIDEO_GROUPS.keys()),
            "environment": "Cloud Run" if os.environ.get('K_SERVICE') else "Local",
            "timestamp": datetime.utcnow().isoformat()
        }
        return debug_info
    except Exception as e:
        logger.error(f"Error in debug route: {e}")
        return {"error": str(e)}, 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    try:
        response = make_response(render_template('404.html'), 404)
        return add_cache_headers(response, 'page')
    except Exception as e:
        logger.error(f"Error in 404 handler: {e}")
        return "Page not found", 404

@app.errorhandler(500)
def server_error(error):
    try:
        response = make_response(render_template('500.html'), 500)
        response.headers['Cache-Control'] = 'no-cache'
        return response
    except Exception as e:
        logger.error(f"Error in 500 handler: {e}")
        return "Internal server error", 500

@app.route("/debug-headers")
def debug_headers():
    headers_info = {key: value for key, value in request.headers}
    return jsonify(headers_info)

@app.route("/test-cdn")
def test_cdn_page():
    try:
        # Pass all headers to the template
        headers_info = {key: value for key, value in request.headers}
        response = make_response(render_template('test_cdn.html', headers_info=headers_info))
        return add_cache_headers(response, 'page')
    except Exception as e:
        logger.error(f"Error in test_cdn_page route: {e}")
        return "Internal server error", 500

@app.route("/dashboard")
def dashboard_page():
    """Render the CDN performance dashboard"""
    try:
        response = make_response(render_template('dashboard.html'))
        return add_cache_headers(response, 'page')
    except Exception as e:
        logger.error(f"Error in dashboard_page route: {e}")
        return "Internal server error", 500

@app.route("/test")
def test_page():
    """Render the A/B test page"""
    try:
        response = make_response(render_template('ab_test.html'))
        return add_cache_headers(response, 'page')
    except Exception as e:
        logger.error(f"Error in ab_test_page route: {e}")
        return "Internal server error", 500

# In-memory storage for performance metrics (in production, use a database)
cdn_metrics = {
    'latency_history': [],
    'regional_data': {},
    'total_requests': 0,
    'cache_hits': 0,
    'cache_misses': 0
}

@app.route("/api/dashboard")
def dashboard_api():
    """API endpoint for dashboard data"""
    try:
        # Check if we have any real data
        total_real_requests = PerformanceMetric.query.count()
        
        # If no real data, return simulated data for development
        if total_real_requests == 0:
            # Simulate some metrics data for development
            import random
            
            # Generate some sample data for demonstration
            regions = ['sea', 'us-east', 'us-west', 'eu-west', 'eu-central', 'asia-pacific']
            region_names = {
                'sea': 'Southeast Asia',
                'us-east': 'US East',
                'us-west': 'US West',
                'eu-west': 'Europe West',
                'eu-central': 'Europe Central',
                'asia-pacific': 'Asia Pacific'
            }
            
            # Generate regional performance data
            regional_performance = {}
            for region in regions:
                regional_performance[region_names[region]] = {
                    'latency': random.randint(20, 200),
                    'requests': random.randint(1000, 50000),
                    'cache_hit_ratio': random.randint(70, 95)
                }
            
            # Calculate overall metrics
            total_requests = sum(region['requests'] for region in regional_performance.values())
            avg_latency = sum(region['latency'] for region in regional_performance.values()) / len(regional_performance)
            cache_hit_ratio = sum(region['cache_hit_ratio'] for region in regional_performance.values()) / len(regional_performance)
            
            # Health status (randomly mark some as unhealthy for demonstration)
            health_status = {}
            for region in regions:
                health_status[region_names[region]] = 'healthy' if random.random() > 0.1 else 'degraded'
            
            # Generate sample latency history
            latency_history = []
            now = datetime.utcnow()
            for i in range(50):
                latency_history.append({
                    'timestamp': (now - timedelta(minutes=i*5)).isoformat(),
                    'latency': random.randint(50, 300),
                    'region': random.choice(list(region_names.values()))
                })
            
            data = {
                'average_latency': round(avg_latency, 1),
                'cache_hit_ratio': round(cache_hit_ratio, 1),
                'active_regions': len(regions),
                'total_requests': total_requests,
                'regional_performance': regional_performance,
                'health_status': health_status,
                'latency_history': latency_history
            }
            
            # Debug logging
            logger.info(f"Simulated dashboard data: {data}")
            
            return jsonify(data)
        
        # Get real metrics from database
        from sqlalchemy import func, extract
        
        # Calculate overall metrics
        total_requests = PerformanceMetric.query.count()
        
        # Calculate average latency
        avg_latency_result = db.session.query(func.avg(PerformanceMetric.latency)).filter(PerformanceMetric.latency != None).first()
        avg_latency = float(avg_latency_result[0]) if avg_latency_result[0] else 0
        
        # Calculate cache hit ratio
        cache_hits = PerformanceMetric.query.filter(PerformanceMetric.cache_status.ilike('HIT')).count()
        cache_hit_ratio = (cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Get regional performance data
        regional_data = db.session.query(
            PerformanceMetric.region,
            func.count(PerformanceMetric.id).label('request_count'),
            func.avg(PerformanceMetric.latency).label('avg_latency'),
            func.sum(case((PerformanceMetric.cache_status.ilike('HIT'), 1), else_=0)).label('cache_hits')
        ).filter(
            PerformanceMetric.region != None
        ).group_by(
            PerformanceMetric.region
        ).all()
        
        regional_performance = {}
        health_status = {}
        
        for region_data in regional_data:
            region_name = region_data.region
            requests = region_data.request_count
            avg_lat = float(region_data.avg_latency) if region_data.avg_latency else 0
            hits = region_data.cache_hits or 0
            hit_ratio = (hits / requests * 100) if requests > 0 else 0
            
            regional_performance[region_name] = {
                'latency': round(avg_lat, 1),
                'requests': requests,
                'cache_hit_ratio': round(hit_ratio, 1)
            }
            
            # Simple health check - consider healthy if latency < 500ms
            health_status[region_name] = 'healthy' if avg_lat < 500 else 'degraded'
        
        # Get recent latency history for charting (last 24 hours)
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        recent_metrics = PerformanceMetric.query.filter(
            PerformanceMetric.timestamp >= twenty_four_hours_ago,
            PerformanceMetric.latency != None
        ).order_by(PerformanceMetric.timestamp.desc()).limit(100).all()
        
        latency_history = []
        for metric in recent_metrics:
            latency_history.append({
                'timestamp': metric.timestamp.isoformat(),
                'latency': metric.latency,
                'region': metric.region
            })
        
        data = {
            'average_latency': round(avg_latency, 1),
            'cache_hit_ratio': round(cache_hit_ratio, 1),
            'active_regions': len(regional_performance),
            'total_requests': total_requests,
            'regional_performance': regional_performance,
            'health_status': health_status,
            'latency_history': latency_history
        }
        
        # Debug logging
        logger.info(f"Real dashboard data: {data}")
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in dashboard_api route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/api/collect-metrics", methods=['POST'])
def collect_metrics():
    """API endpoint to collect CDN performance metrics from clients"""
    try:
        data = request.get_json()
        
        # Extract metrics from the request
        latency = data.get('latency')
        region = data.get('region')
        cache_status = data.get('cache_status')
        content_type = data.get('content_type')
        
        # Store metrics in database
        metric = PerformanceMetric(
            latency=latency,
            region=region,
            cache_status=cache_status,
            content_type=content_type,
            user_agent=request.headers.get('User-Agent', ''),
            ip_address=request.remote_addr
        )
        db.session.add(metric)
        db.session.commit()
        
        # Update our in-memory metrics store
        if latency:
            cdn_metrics['latency_history'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'latency': latency,
                'region': region
            })
            
            # Keep only the last 1000 entries
            if len(cdn_metrics['latency_history']) > 1000:
                cdn_metrics['latency_history'] = cdn_metrics['latency_history'][-1000:]
        
        if cache_status:
            if cache_status.upper() == 'HIT':
                cdn_metrics['cache_hits'] += 1
            else:
                cdn_metrics['cache_misses'] += 1
        
        cdn_metrics['total_requests'] += 1
        
        # Update regional data
        if region:
            if region not in cdn_metrics['regional_data']:
                cdn_metrics['regional_data'][region] = {
                    'requests': 0,
                    'total_latency': 0,
                    'cache_hits': 0,
                    'cache_misses': 0
                }
            
            cdn_metrics['regional_data'][region]['requests'] += 1
            if latency:
                cdn_metrics['regional_data'][region]['total_latency'] += latency
            if cache_status:
                if cache_status.upper() == 'HIT':
                    cdn_metrics['regional_data'][region]['cache_hits'] += 1
                else:
                    cdn_metrics['regional_data'][region]['cache_misses'] += 1
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error in collect_metrics route: {e}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route("/api/generate-test-data", methods=['POST'])
def generate_test_data():
    """API endpoint to generate test data for development"""
    try:
        import random
        from datetime import timedelta
        
        # Generate 50 test metrics
        test_regions = ['sea', 'us-east', 'us-west', 'eu-west', 'eu-central', 'asia-pacific', 'local-dev']
        cache_statuses = ['HIT', 'MISS', 'EXPIRED']
        content_types = ['page', 'image', 'video']
        
        for _ in range(50):
            metric = PerformanceMetric(
                latency=random.randint(50, 500),
                region=random.choice(test_regions),
                cache_status=random.choice(cache_statuses),
                content_type=random.choice(content_types),
                user_agent='Test Agent',
                ip_address='127.0.0.1',
                timestamp=datetime.utcnow() - timedelta(minutes=random.randint(0, 1440))  # Up to 24 hours ago
            )
            db.session.add(metric)
        
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Generated 50 test metrics'})
    except Exception as e:
        logger.error(f"Error in generate_test_data route: {e}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500




@app.route("/test/run/<region>")
def run_test(region):
    """Triggers the A/B test script in the background."""
    if region not in ['sea', 'eu', 'us']:
        return jsonify({"status": "error", "message": "Invalid region"}), 400

    try:
        # Open a log file to capture the output
        log_file = open("ab_test.log", "w")
        # Start the script in a new process
        subprocess.Popen(["python3", "run_ab_test.py", region], stdout=log_file, stderr=subprocess.STDOUT)
        return jsonify({"status": "success", "message": f"A/B test started for {region} region. See ab_test.log for progress."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/test/status")
def test_status():
    """Returns the content of the A/B test log file."""
    try:
        with open("ab_test.log", "r") as f:
            content = f.read()
        return jsonify({"status": "success", "log": content})
    except FileNotFoundError:
        return jsonify({"status": "pending", "log": "No test has been run yet."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    try:
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully")
        start_cdn_healthcheck()
        app.run(debug=True, host="0.0.0.0", port=8080)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise