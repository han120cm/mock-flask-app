#!/bin/bash

# Prometheus Installation Script for nginx CDN VM
# Run this as root or with sudo

set -e

echo "🚀 Starting Prometheus installation..."

# Create prometheus user
echo "📝 Creating prometheus user..."
useradd --no-create-home --shell /bin/false prometheus

# Create directories
echo "📁 Creating directories..."
mkdir -p /etc/prometheus
mkdir -p /var/lib/prometheus
chown prometheus:prometheus /var/lib/prometheus

# Download Prometheus (adjust version as needed)
PROMETHEUS_VERSION="3.4.2"
echo "⬇️  Downloading Prometheus v${PROMETHEUS_VERSION}..."
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz
tar -xzf prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz

# Copy binaries
echo "📋 Copying Prometheus binaries..."
cp prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus /usr/local/bin/
cp prometheus-${PROMETHEUS_VERSION}.linux-amd64/promtool /usr/local/bin/
chown prometheus:prometheus /usr/local/bin/prometheus
chown prometheus:prometheus /usr/local/bin/promtool

# Copy configuration
echo "⚙️  Setting up Prometheus configuration..."
cp prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus.yml /etc/prometheus/
chown prometheus:prometheus /etc/prometheus/prometheus.yml

# Clean up
rm -rf prometheus-${PROMETHEUS_VERSION}.linux-amd64*

echo "✅ Prometheus installation completed!"
echo "📊 Prometheus will be available at: http://your-vm-ip:9090" 