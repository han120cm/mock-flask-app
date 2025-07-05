#!/bin/bash

# Phase 1: Complete Infrastructure Setup Script
# Run this as root or with sudo

set -e

echo "🎯 Starting Phase 1: Infrastructure Setup for nginx CDN Monitoring"
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root or with sudo"
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install required dependencies
print_status "Installing dependencies..."
apt install -y wget curl systemd

# Step 1: Install Prometheus
print_status "Step 1/3: Installing Prometheus..."
bash install-prometheus.sh

# Copy configuration and service files
cp prometheus.yml /etc/prometheus/
cp prometheus.service /etc/systemd/system/
chown prometheus:prometheus /etc/prometheus/prometheus.yml

# Step 2: Install Node Exporter
print_status "Step 2/3: Installing Node Exporter..."
bash install-node-exporter.sh

# Copy service file
cp node-exporter.service /etc/systemd/system/

# Step 3: Install nginx Prometheus Exporter
print_status "Step 3/3: Installing nginx Prometheus Exporter..."
bash install-nginx-exporter.sh

# Copy service file
cp nginx-exporter.service /etc/systemd/system/

# Reload systemd and enable services
print_status "Enabling and starting services..."
systemctl daemon-reload

systemctl enable prometheus
systemctl start prometheus

systemctl enable node_exporter
systemctl start node_exporter

systemctl enable nginx-exporter
systemctl start nginx-exporter

# Wait a moment for services to start
sleep 5

# Verify services are running
print_status "Verifying services..."
services=("prometheus" "node_exporter" "nginx-exporter")

for service in "${services[@]}"; do
    if systemctl is-active --quiet $service; then
        print_status "$service is running ✅"
    else
        print_error "$service failed to start ❌"
        systemctl status $service
    fi
done

# Check if ports are listening
print_status "Checking if ports are listening..."
netstat -tlnp | grep -E ':(9090|9100|9113)'

echo ""
echo "🎉 Phase 1 Installation Complete!"
echo "=================================="
echo "📊 Prometheus: http://$(hostname -I | awk '{print $1}'):9090"
echo "📈 Node Exporter: http://$(hostname -I | awk '{print $1}'):9100/metrics"
echo "🌐 nginx Exporter: http://$(hostname -I | awk '{print $1}'):9113/metrics"
echo ""
print_warning "Note: nginx-exporter requires nginx status module to be enabled in Phase 2"
echo ""
echo "Next steps:"
echo "1. Verify all services are running: systemctl status prometheus node_exporter nginx-exporter"
echo "2. Check Prometheus targets: http://$(hostname -I | awk '{print $1}'):9090/targets"
echo "3. Proceed to Phase 2: nginx Configuration Enhancements" 