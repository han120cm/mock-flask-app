#!/bin/bash

# nginx Prometheus Exporter Installation Script
# Run this as root or with sudo

set -e

echo "🚀 Starting nginx Prometheus Exporter installation..."

# Create nginx_exporter user
echo "📝 Creating nginx_exporter user..."
useradd --no-create-home --shell /bin/false nginx_exporter

# Download nginx Prometheus Exporter
NGINX_EXPORTER_VERSION="0.11.0"
echo "⬇️  Downloading nginx Prometheus Exporter v${NGINX_EXPORTER_VERSION}..."
cd /tmp
wget https://github.com/nginxinc/nginx-prometheus-exporter/releases/download/v${NGINX_EXPORTER_VERSION}/nginx-prometheus-exporter_${NGINX_EXPORTER_VERSION}_linux_amd64.tar.gz
tar -xzf nginx-prometheus-exporter_${NGINX_EXPORTER_VERSION}_linux_amd64.tar.gz

# Copy binary
echo "📋 Copying nginx Prometheus Exporter binary..."
cp nginx-prometheus-exporter /usr/local/bin/
chown nginx_exporter:nginx_exporter /usr/local/bin/nginx-prometheus-exporter

# Clean up
rm nginx-prometheus-exporter_${NGINX_EXPORTER_VERSION}_linux_amd64.tar.gz
rm nginx-prometheus-exporter

echo "✅ nginx Prometheus Exporter installation completed!"
echo "📊 nginx Exporter will be available at: http://your-vm-ip:9113/metrics" 