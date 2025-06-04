from flask import Flask, render_template, url_for, request, make_response
from datetime import datetime, timedelta
import hashlib

app = Flask(__name__)

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

def make_urls(base_url, start, end, region='default'):
    """Generate URLs - always use CDN, let GeoDNS handle routing"""
    # Always use the same CDN domain - GeoDNS will route to nearest server
    cdn_url = base_url.replace('https://storage.googleapis.com/bucket-main-ta', CDN_ENDPOINT)
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
    
    # Get CDN region from the header set by NGINX
    cdn_region = request.headers.get('X-CDN-Region', 'Unknown')
    
    # Set flag based on region
    if cdn_region == 'EU-West':
        cdn_flag = '🇪🇺'
    elif 'US' in cdn_region:
        cdn_flag = '🇺🇸'
    else:
        cdn_flag = '🌐'
        
    start, end = IMAGE_GROUPS[group]
    
    # Always use CDN - GeoDNS routes to nearest server
    IMAGE_BASE = "https://storage.googleapis.com/bucket-main-ta/static/images/image_{}.jpg"
    urls = make_urls(IMAGE_BASE, start, end)
    
    response = make_response(render_template('images.html', group=group, urls=urls, CDN_REGION=cdn_region, CDN_FLAG=cdn_flag))
    return add_cache_headers(response, 'page')

@app.route("/videos/<group>")
def show_videos(group):
    if group not in VIDEO_GROUPS:
        return "Invalid video group", 404
    
    # Get CDN region from the header set by NGINX
    cdn_region = request.headers.get('X-CDN-Region', 'Unknown')
    
    # Set flag based on region
    if cdn_region == 'EU-West':
        cdn_flag = '🇪🇺'
    elif 'US' in cdn_region:
        cdn_flag = '🇺🇸'
    else:
        cdn_flag = '🌐'
        
    start, end = VIDEO_GROUPS[group]
    
    VIDEO_BASE = "https://storage.googleapis.com/bucket-main-ta/static/videos/video_{}.mp4"
    urls = make_urls(VIDEO_BASE, start, end)

    response = make_response(render_template('videos.html', group=group, urls=urls, CDN_REGION=cdn_region, CDN_FLAG=cdn_flag))
    return add_cache_headers(response, 'page')

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
    app.run(debug=True, host="0.0.0.0", port=8000)