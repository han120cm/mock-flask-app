#!/bin/bash
# Script to install k6 on Debian/Ubuntu systems.

set -e

echo "Updating package list..."
sudo apt-get update

echo "Installing prerequisite packages..."
sudo apt-get install -y debian-keyring apt-transport-https

echo "Importing k6 GPG key..."
sudo gpg -o /usr/share/keyrings/k6-archive-keyring.gpg --no-default-keyring --keyring /usr/share/keyrings/debian-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69

echo "Adding k6 repository..."
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list

echo "Updating package list again with k6 repo..."
sudo apt-get update

echo "Installing k6..."
sudo apt-get install -y k6

echo "k6 installation complete. Verify with 'k6 version'."
