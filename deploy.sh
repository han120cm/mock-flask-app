#!/bin/bash

# Deploy to Cloud Run
# Make sure you have gcloud CLI installed and configured

set -e

echo "🚀 Deploying to Cloud Run..."

# Get the current project ID
PROJECT_ID=$(gcloud config get-value project)
echo "📋 Project ID: $PROJECT_ID"

# Build and push the Docker image
echo "🔨 Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/mock-flask-app .

echo "📤 Pushing to Container Registry..."
docker push gcr.io/$PROJECT_ID/mock-flask-app

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy mock-flask-app \
  --image gcr.io/$PROJECT_ID/mock-flask-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 10 \
  --timeout 300

echo "✅ Deployment complete!"
echo "🌐 Your app is available at:"
gcloud run services describe mock-flask-app --platform managed --region us-central1 --format="value(status.url)" 