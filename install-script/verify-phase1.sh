#!/bin/bash

# Phase 1 Verification Script
# Run this to verify all components are working

echo "🔍 Phase 1 Verification Script"
echo "=============================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_ok() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if services are running
echo "1. Checking service status..."
services=("prometheus" "node-exporter" "nginx-exporter")

for service in "${services[@]}"; do
    if systemctl is-active --quiet $service; then
        print_ok "$service is running"
    else
        print_error "$service is not running"
        systemctl status $service --no-pager -l
    fi
done

echo ""
echo "2. Checking if ports are listening..."
ports=("9090:Prometheus" "9100:Node Exporter" "9113:nginx Exporter")

for port_info in "${ports[@]}"; do
    port=$(echo $port_info | cut -d: -f1)
    service_name=$(echo $port_info | cut -d: -f2)
    
    if netstat -tlnp | grep -q ":$port "; then
        print_ok "$service_name is listening on port $port"
    else
        print_error "$service_name is not listening on port $port"
    fi
done

echo ""
echo "3. Testing HTTP endpoints..."

# Test Prometheus
if curl -s http://localhost:9090 > /dev/null; then
    print_ok "Prometheus web interface is accessible"
else
    print_error "Prometheus web interface is not accessible"
fi

# Test Node Exporter
if curl -s http://localhost:9100/metrics | grep -q "node_"; then
    print_ok "Node Exporter metrics are available"
else
    print_error "Node Exporter metrics are not available"
fi

# Test nginx Exporter
if curl -s http://localhost:9113/metrics > /dev/null; then
    print_ok "nginx Exporter metrics endpoint is accessible"
else
    print_warning "nginx Exporter metrics endpoint is not accessible (may need nginx status module)"
fi

echo ""
echo "4. Checking Prometheus targets..."
if curl -s http://localhost:9090/api/v1/targets | grep -q "up"; then
    print_ok "Prometheus has some targets in UP state"
else
    print_warning "No Prometheus targets are UP (this is normal if nginx status is not configured yet)"
fi

echo ""
echo "5. System resource usage..."
echo "Memory usage:"
free -h | grep -E "Mem|Swap"

echo ""
echo "Disk usage:"
df -h | grep -E "/$|/var"

echo ""
echo "🎯 Verification Summary:"
echo "======================="
echo "If you see mostly ✅ marks above, Phase 1 is successful!"
echo ""
echo "📊 Access URLs:"
echo "- Prometheus: http://$(hostname -I | awk '{print $1}'):9090"
echo "- Node Exporter: http://$(hostname -I | awk '{print $1}'):9100/metrics"
echo "- nginx Exporter: http://$(hostname -I | awk '{print $1}'):9113/metrics"
echo ""
print_warning "Note: nginx-exporter may show warnings until Phase 2 is completed"
echo ""
echo "Ready for Phase 2: nginx Configuration Enhancements" 