# Phase 1: Infrastructure Setup - nginx CDN Monitoring

This phase sets up the core monitoring infrastructure for your nginx CDN VM using Prometheus and related exporters.

## 🎯 What We're Installing

### 1. **Prometheus** (Port 9090)
- **Purpose**: Time-series database that collects and stores metrics
- **What it monitors**: All system and application metrics
- **Access**: `http://your-vm-ip:9090`

### 2. **Node Exporter** (Port 9100)
- **Purpose**: Collects system-level metrics
- **What it monitors**: CPU, RAM, disk I/O, network, processes
- **Metrics**: `http://your-vm-ip:9100/metrics`

### 3. **nginx Prometheus Exporter** (Port 9113)
- **Purpose**: Converts nginx metrics to Prometheus format
- **What it monitors**: nginx status, access logs, cache performance
- **Metrics**: `http://your-vm-ip:9113/metrics`

## 📁 Files Created

```
mock-flask-app/
├── install-prometheus.sh          # Prometheus installation script
├── install-node-exporter.sh       # Node Exporter installation script
├── install-nginx-exporter.sh      # nginx Exporter installation script
├── phase1-install.sh              # Master installation script
├── verify-phase1.sh               # Verification script
├── prometheus.yml                 # Prometheus configuration
├── prometheus.service             # Prometheus systemd service
├── node-exporter.service          # Node Exporter systemd service
├── nginx-exporter.service         # nginx Exporter systemd service
└── PHASE1-README.md              # This file
```

## 🚀 Quick Installation

### Option 1: Automated Installation (Recommended)
```bash
# Copy all files to your VM
# Then run the master script
sudo bash phase1-install.sh
```

### Option 2: Manual Installation
```bash
# 1. Install Prometheus
sudo bash install-prometheus.sh
sudo cp prometheus.yml /etc/prometheus/
sudo cp prometheus.service /etc/systemd/system/

# 2. Install Node Exporter
sudo bash install-node-exporter.sh
sudo cp node-exporter.service /etc/systemd/system/

# 3. Install nginx Exporter
sudo bash install-nginx-exporter.sh
sudo cp nginx-exporter.service /etc/systemd/system/

# 4. Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable prometheus node_exporter nginx-exporter
sudo systemctl start prometheus node_exporter nginx-exporter
```

## ✅ Verification

After installation, run the verification script:
```bash
bash verify-phase1.sh
```

You should see:
- ✅ All services running
- ✅ All ports listening
- ✅ HTTP endpoints accessible
- ⚠️ nginx-exporter may show warnings (normal until Phase 2)

## 📊 What You'll See

### Prometheus Dashboard
- **URL**: `http://your-vm-ip:9090`
- **Features**: 
  - Query interface for metrics
  - Target status page
  - Basic graphs
  - Alert manager (Phase 4)

### Available Metrics (after Phase 2)
- **System**: CPU, memory, disk usage
- **Network**: Bandwidth, connections
- **nginx**: Requests, response codes, cache hits
- **CDN**: Cache performance, geographic distribution

## 🔧 Configuration Details

### Prometheus Configuration (`prometheus.yml`)
- Scrapes metrics every 15 seconds
- Monitors localhost targets
- Stores data in `/var/lib/prometheus/`

### Service Configuration
- All services run as dedicated users
- Automatic restart on failure
- Logs available via `journalctl`

## 🚨 Important Notes

1. **Firewall**: Ensure ports 9090, 9100, 9113 are open
2. **Security**: Services are configured for local access only
3. **Dependencies**: nginx-exporter requires nginx status module (Phase 2)
4. **Storage**: Prometheus data stored in `/var/lib/prometheus/`

## 🔍 Troubleshooting

### Service Not Starting
```bash
# Check service status
sudo systemctl status prometheus
sudo journalctl -u prometheus -f

# Check logs
sudo journalctl -u node_exporter -f
sudo journalctl -u nginx-exporter -f
```

### Port Not Listening
```bash
# Check if ports are open
sudo netstat -tlnp | grep -E ':(9090|9100|9113)'

# Check firewall
sudo ufw status
```

### Metrics Not Available
```bash
# Test endpoints manually
curl http://localhost:9090
curl http://localhost:9100/metrics
curl http://localhost:9113/metrics
```

## 📈 Next Steps

After Phase 1 completion:
1. ✅ Verify all services are running
2. 🔄 Proceed to **Phase 2: nginx Configuration Enhancements**
3. 📊 Set up Grafana dashboards (Phase 3)
4. 🚨 Configure alerting (Phase 4)

## 🎯 Success Criteria

Phase 1 is successful when:
- ✅ Prometheus is accessible at port 9090
- ✅ Node Exporter provides system metrics
- ✅ All services start automatically on boot
- ✅ No critical errors in service logs
- ✅ Ready for Phase 2 nginx configuration

---

**Ready for Phase 2?** Let's enhance your nginx configuration to expose detailed CDN metrics! 🚀 