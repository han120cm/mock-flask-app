# Troubleshooting Guide for 500 Errors

This guide will help you diagnose and fix 500 errors in your Cloud Run deployment.

## Quick Diagnosis

### 1. Check the Debug Endpoint

First, check the debug endpoint to see what's working:

```bash
curl https://web-server-577176926733.us-central1.run.app/debug
```

This will show you:
- Database path
- Upload folder
- GCS availability
- CDN availability
- Environment detection

### 2. Check Cloud Run Logs

View the logs to see what's causing the 500 error:

```bash
gcloud logs read --service=web-server --limit=50
```

Or stream logs in real-time:

```bash
gcloud logs tail --service=web-server
```

### 3. Test Individual Routes

Use the test script to check specific routes:

```bash
python test_routes.py https://web-server-577176926733.us-central1.run.app
```

## Common Issues and Solutions

### Issue 1: Database Connection Problems

**Symptoms:**
- 500 error on routes that query the database
- Logs show database-related errors

**Solutions:**

1. **Check database path:**
   ```bash
   curl https://web-server-577176926733.us-central1.run.app/debug
   ```
   The database path should be `/tmp/site.db` on Cloud Run.

2. **Verify database permissions:**
   The `/tmp` directory should be writable.

3. **Recreate database:**
   ```bash
   # Deploy with fresh database
   gcloud run deploy web-server --image gcr.io/YOUR_PROJECT_ID/mock-flask-app --platform managed --region us-central1
   ```

### Issue 2: Google Cloud Storage Problems

**Symptoms:**
- 500 error on upload routes
- Logs show GCS-related errors

**Solutions:**

1. **Check GCS availability:**
   ```bash
   curl https://web-server-577176926733.us-central1.run.app/debug
   ```
   Look for `"gcs_available": true`

2. **Set up service account permissions:**
   ```bash
   # Get your project number
   PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")
   
   # Grant Storage Admin role to Cloud Run service account
   gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
     --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
     --role="roles/storage.admin"
   ```

3. **Set up credentials:**
   ```bash
   # Deploy with service account
   gcloud run deploy web-server \
     --image gcr.io/YOUR_PROJECT_ID/mock-flask-app \
     --platform managed \
     --region us-central1 \
     --service-account=YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com
   ```

### Issue 3: Template Rendering Problems

**Symptoms:**
- 500 error on all routes
- Logs show template-related errors

**Solutions:**

1. **Check template files:**
   Make sure all templates exist:
   - `templates/base.html`
   - `templates/home.html`
   - `templates/images.html`
   - `templates/videos.html`
   - `templates/404.html`
   - `templates/500.html`

2. **Verify template syntax:**
   Check for syntax errors in templates.

### Issue 4: CDN Healthcheck Problems

**Symptoms:**
- 500 error on routes that use CDN
- Logs show CDN-related errors

**Solutions:**

1. **Check CDN availability:**
   ```bash
   curl https://web-server-577176926733.us-central1.run.app/debug
   ```
   Look for `"cdn_available": true`

2. **Disable CDN temporarily:**
   The app will fallback to original URLs if CDN is unavailable.

### Issue 5: Memory/Resource Issues

**Symptoms:**
- Intermittent 500 errors
- Timeout errors

**Solutions:**

1. **Increase memory:**
   ```bash
   gcloud run deploy web-server \
     --image gcr.io/YOUR_PROJECT_ID/mock-flask-app \
     --platform managed \
     --region us-central1 \
     --memory 1Gi
   ```

2. **Increase timeout:**
   ```bash
   gcloud run deploy web-server \
     --image gcr.io/YOUR_PROJECT_ID/mock-flask-app \
     --platform managed \
     --region us-central1 \
     --timeout 300
   ```

## Step-by-Step Debugging

### Step 1: Check Basic Health

```bash
curl https://web-server-577176926733.us-central1.run.app/health
```

Expected response:
```json
{"status": "healthy", "timestamp": "..."}
```

### Step 2: Check Debug Information

```bash
curl https://web-server-577176926733.us-central1.run.app/debug
```

Look for:
- `"environment": "Cloud Run"`
- `"gcs_available": true`
- `"cdn_available": true`

### Step 3: Test Home Page

```bash
curl https://web-server-577176926733.us-central1.run.app/
```

Should return HTML content.

### Step 4: Test Image Routes

```bash
curl https://web-server-577176926733.us-central1.run.app/images/trending
```

Should return HTML content.

### Step 5: Check Logs

```bash
gcloud logs read --service=web-server --limit=20
```

Look for:
- Error messages
- Stack traces
- Database errors
- Template errors

## Quick Fixes

### Fix 1: Redeploy with Fresh Database

```bash
# Redeploy the service
gcloud run deploy web-server \
  --image gcr.io/YOUR_PROJECT_ID/mock-flask-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Fix 2: Set Environment Variables

```bash
gcloud run deploy web-server \
  --image gcr.io/YOUR_PROJECT_ID/mock-flask-app \
  --platform managed \
  --region us-central1 \
  --set-env-vars FLASK_ENV=production
```

### Fix 3: Increase Resources

```bash
gcloud run deploy web-server \
  --image gcr.io/YOUR_PROJECT_ID/mock-flask-app \
  --platform managed \
  --region us-central1 \
  --memory 1Gi \
  --cpu 2 \
  --timeout 300
```

## Monitoring and Alerts

### Set up Monitoring

1. **Enable Cloud Monitoring:**
   ```bash
   gcloud services enable monitoring.googleapis.com
   ```

2. **Create alerts for 500 errors:**
   - Go to Cloud Console > Monitoring
   - Create alerting policy for HTTP 500 errors

### View Metrics

1. **Cloud Run metrics:**
   - Request count
   - Error rate
   - Response time
   - Memory usage

2. **Application metrics:**
   - Database connections
   - GCS operations
   - CDN health

## Prevention

### Best Practices

1. **Use proper error handling:**
   - All routes have try-catch blocks
   - Log errors with context
   - Return appropriate HTTP status codes

2. **Test before deployment:**
   - Run tests locally
   - Use the test scripts provided
   - Check all routes

3. **Monitor continuously:**
   - Set up alerts
   - Check logs regularly
   - Monitor performance

4. **Use proper resource limits:**
   - Set appropriate memory and CPU
   - Configure timeouts
   - Use auto-scaling

## Getting Help

If you're still experiencing issues:

1. **Check the logs:**
   ```bash
   gcloud logs read --service=web-server --limit=100
   ```

2. **Run the verification script:**
   ```bash
   python verify_deployment.py https://web-server-577176926733.us-central1.run.app
   ```

3. **Test locally:**
   ```bash
   python app.py
   ```

4. **Check the debug endpoint:**
   ```bash
   curl https://web-server-577176926733.us-central1.run.app/debug
   ``` 