#!/bin/bash

# Node Exporter Installation Script
# Run this as root or with sudo

set -e

echo "🚀 Starting Node Exporter installation..."

# Create node_exporter user
echo "📝 Creating node_exporter user..."
useradd --no-create-home --shell /bin/false node_exporter

# Download Node Exporter
NODE_EXPORTER_VERSION="1.6.1"
echo "⬇️  Downloading Node Exporter v${NODE_EXPORTER_VERSION}..."
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
tar -xzf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz

# Copy binary
echo "📋 Copying Node Exporter binary..."
cp node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64/node_exporter /usr/local/bin/
chown node_exporter:node_exporter /usr/local/bin/node_exporter

# Clean up
rm -rf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64*

echo "✅ Node Exporter installation completed!"
echo "📊 Node Exporter will be available at: http://your-vm-ip:9100/metrics" 