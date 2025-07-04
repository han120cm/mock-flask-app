# Mock Flask App - CDN Testbed

This is a testbed application for demonstrating CDN (Content Delivery Network) functionality for a thesis project. The app shows how content can be served from different CDN regions and displays which region is serving the content to users.

## Features

- **CDN Region Detection**: Automatically detects and displays which CDN region is serving content
- **Media Archive**: Browse images and videos organized by categories
- **Real-time CDN Info**: Shows current CDN region with flag emoji and descriptive name
- **Debug Tools**: Built-in debugging endpoints to verify CDN functionality

## CDN Configuration

The app is designed to work with an nginx reverse proxy that sets custom headers:

- `X-CDN-Region`: The CDN region code (e.g., "sea" for Southeast Asia)
- `X-User-Region`: User's detected region
- `X-Cache-Status`: Cache hit/miss status
- `X-Served-From`: Source of the content

## Running the Application

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Flask app:
   ```bash
   python app.py
   ```

3. Access the application:
   - Home page: http://localhost:8000/
   - Test CDN: http://localhost:8000/test-cdn
   - Debug headers: http://localhost:8000/debug-headers
  
## Config Remote CDN
```bash
# Update 
sudo apt update

# Install nginx
sudo apt install nginx

# Check nginx status
sudo systemctl status nginx

# Set nginx configuration
sudo nano /etc/nginx/nginx.conf

# copy paste the nginx.conf from repo

# Set server block configuration
sudo nano /etc/nginx/conf.d/cdn.conf

# Set SSL certificate using LetsEncrypt
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d cdn.sohryuu.me

# Notes: for the root domain (e.g. sohryuu.me) use tutorial from the domain registrar

# copy paste the cdn.conf from repo

# Make cache directory
sudo mkdir -p /var/cache/nginx
sudo chown -R www-data:www-data /var/cache/nginx  # For Ubuntu/Debian

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```


## Testing CDN Functionality

### Local Testing (with mock headers)
```bash
# Test with Southeast Asia region
curl -H "X-CDN-Region: sea" http://localhost:8000/test-cdn

# Test with US East region
curl -H "X-CDN-Region: us-east" http://localhost:8000/test-cdn

# Test with Europe region
curl -H "X-CDN-Region: eu-west" http://localhost:8000/test-cdn
```

### Production Testing
When deployed behind your nginx CDN proxy:
1. Visit https://cdn.sohryuu.me/test-cdn
2. The page should show "🇮🇩 Southeast Asia" as the CDN region
3. Check the debug headers for detailed information

## CDN Regions Supported

- `sea` → 🇮🇩 Southeast Asia
- `us-east` → 🇺🇸 US East Coast  
- `us-west` → 🇺🇸 US West Coast
- `eu-west` → 🇪🇺 Europe West
- `eu-central` → 🇪🇺 Europe Central
- `asia-pacific` → 🌏 Asia Pacific

## File Structure

```
mock-flask-app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── static/
│   └── css/
│       └── styles.css     # Styling for the application
└── templates/
    ├── base.html          # Base template
    ├── home.html          # Home page
    ├── images.html        # Images page
    ├── videos.html        # Videos page
    ├── test_cdn.html      # CDN test page
    └── 404.html           # 404 error page
```

## Nginx Configuration

The app is designed to work with the provided nginx configuration (`cdn.conf`) that:
- Sets custom headers for CDN region identification
- Implements caching for media and page content
- Provides rate limiting
- Handles SSL termination

## Debugging

Use the `/debug-headers` endpoint to see all request headers and CDN information in JSON format.

Use the `/test-cdn` endpoint for a user-friendly interface to verify CDN functionality.
