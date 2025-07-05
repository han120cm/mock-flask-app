# Mock Flask App - CDN Testbed

This is a testbed application for demonstrating CDN (Content Delivery Network) functionality for a thesis project. The app shows how content can be served from different CDN regions and displays which region is serving the content to users.

## Features

- **CDN Region Detection**: Automatically detects and displays which CDN region is serving content (using GCore GeoDNS)
- **Media Archive**: Browse images and videos organized by categories
- **Real-time CDN Info**: Shows current CDN region with flag emoji and descriptive name ( in progress )
- **Debug Tools**: Built-in debugging endpoints to verify CDN functionality ( in progress ) 

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
### Monitor Implementation 
1. Install Prometheus with install-prometheus.sh
2. Create /etc/prometheus/prometheus.yml
3. Create systemd service for prometheus (prometheus.service) in /etc/systemd/system/prometheus.service
4. Enable service :
```bash
sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
sudo systemctl status prometheus
```
5. Install node exporter with install-node-exporter.sh script
6. Create systemd service (node-exporter.service) in /etc/systemd/system/node-exporter.service
7. Enable service :
```bash
sudo systemctl daemon-reload
sudo systemctl enable node-exporter
sudo systemctl start node-exporter
sudo systemctl status node-exporter
```
8. Install nginx prometheus exporter with install-nginx-exporter.sh script
9. Create systemd service (nginx-exporter.service) in /etc/systemd/system/nginx-exporter.service
10. Enable service
```bash
sudo systemctl daemon-reload
sudo systemctl enable nginx-exporter
sudo systemctl start nginx-exporter
sudo systemctl status nginx-exporter
```
11. Verify set up with running phase1-install.sh script ( if error was seen, install net-tools)
12. Enable nginx status module
```bash
   location /nginx_status {
       stub_status;
       allow 127.0.0.1;        # Only allow local access (for security)
       deny all;
   }
```
13. Run test
```bash
sudo nginx -t
sudo systemctl reload nginx
curl http://localhost/nginx_status

# Expected something like this
   Active connections: 1
   server accepts handled requests
    10 10 10
   Reading: 0 Writing: 1 Waiting: 0
```
14. Improve nginx access loggin for caching metrics
```bash
   log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                     '$status $body_bytes_sent "$http_referer" '
                     '"$http_user_agent" "$http_x_forwarded_for" '
                     'cache_status=$upstream_cache_status '
                     'cdn_region=$http_x_cdn_region '
                     'rt=$request_time';
   access_log  /var/log/nginx/access.log  main;


   # After modifying
   sudo systemctl reload nginx
```
15. Verify Prometheus scraping
```bash
curl http://localhost:9113/metrics 
```

### Grafana Dashboard Set Up
1. Install Grafana
```bash
sudo apt-get install -y apt-transport-https software-properties-common wget
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install -y grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```
2. Access Grafana at http://ip-address:3000
3. Add Prometheus as data source (Configuration -> Data Sources), set URL to http://localhost:9090 -> Save and Test
4. Add Firewall (VPC in GCP) to allow port used by Grafana and Prometheus

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
