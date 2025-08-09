#!/bin/bash
# Script to build and run k6 load tests in multiple GCP regions using on-demand Cloud Run Jobs.

set -e

# --- Configuration ---
# REQUIRED: Your Google Cloud Project ID
GCP_PROJECT_ID="tugas-akhir-458309"

# REQUIRED: The GCP region for your Artifact Registry (e.g., "us-central1")
ARTIFACT_REGISTRY_REGION="asia-southeast1"

# REQUIRED: A name for your Artifact Registry repository
ARTIFACT_REPO_NAME="cdn-test-runners"

# The name for the Docker image
IMAGE_NAME="k6-geo-runner"

# The target URL for the load test
TARGET_URL="https://cdn.sohryuu.me"

# The k6 script to execute inside the container
TEST_SCRIPT="load-test-ver2.js"

# List of GCP regions to run the tests from.
TEST_REGIONS=(
  "asia-east1"
  "asia-northeast1"
  "asia-southeast1"
)

# --- Script ---

# Validate configuration
if [ "$GCP_PROJECT_ID" == "your-gcp-project-id" ]; then
    echo "ERROR: Please edit this script and set your GCP_PROJECT_ID and other configuration variables."
    exit 1
fi

# Validate required files exist
if [ ! -f "testing/Dockerfile" ]; then
    echo "ERROR: testing/Dockerfile not found. Please ensure the Dockerfile exists."
    exit 1
fi

if [ ! -f "testing/$TEST_SCRIPT" ]; then
    echo "ERROR: testing/$TEST_SCRIPT not found. Please ensure the k6 test script exists."
    exit 1
fi

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')][INFO] $1"
}

# Function to extract performance metrics from logs
extract_metrics() {
    local log_content="$1"
    local region="$2"
    
    echo ""
    echo "=== PERFORMANCE METRICS FOR $region ==="
    
    # Extract video performance metrics
    echo "$log_content" | grep -E "VIDEO (SUCCESS|ERROR)" | head -20
    
    # Extract image performance metrics  
    echo "$log_content" | grep -E "IMAGE (SUCCESS|ERROR)" | head -20
    
    # Extract summary metrics
    echo "$log_content" | grep -A 20 "CDN Performance Test Results"
    
    # Extract k6 built-in metrics
    echo "$log_content" | grep -E "(http_req_duration|image_req_duration|video_req_duration|cache_hit_rate)" || echo "No k6 metrics found"
    
    echo "======================================="
    echo ""
}

# Set the project for gcloud commands
gcloud config set project $GCP_PROJECT_ID

# 1. Set up Artifact Registry
log "Checking for Artifact Registry repository '$ARTIFACT_REPO_NAME'..."
if ! gcloud artifacts repositories describe $ARTIFACT_REPO_NAME --location=$ARTIFACT_REGISTRY_REGION >/dev/null 2>&1; then
    log "Repository not found. Creating Artifact Registry repository '$ARTIFACT_REPO_NAME'..."
    gcloud artifacts repositories create $ARTIFACT_REPO_NAME \
        --repository-format=docker \
        --location=$ARTIFACT_REGISTRY_REGION \
        --description="Repository for k6 test runners"
else
    log "Artifact Registry repository already exists."
fi

# Configure Docker to authenticate with Artifact Registry
log "Configuring Docker authentication for Artifact Registry..."
gcloud auth configure-docker $ARTIFACT_REGISTRY_REGION-docker.pkg.dev

# Define the full image path
FULL_IMAGE_PATH="$ARTIFACT_REGISTRY_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REPO_NAME/$IMAGE_NAME:latest"

# 2. Build and Push the Docker Image
log "Building the k6 Docker image for linux/amd64 platform..."
docker build --platform linux/amd64 -t $FULL_IMAGE_PATH -f testing/Dockerfile testing/

log "Pushing image to Artifact Registry: $FULL_IMAGE_PATH"
docker push $FULL_IMAGE_PATH

# 3. Deploy Jobs First (without executing)
log "Deploying Cloud Run jobs across regions..."

# This file will store the job info for each region
JOBS_FILE=$(mktemp)

for region in "${TEST_REGIONS[@]}"; do
    JOB_NAME="k6-test-job-${region}-$(date +%s)" # Unique job name with proper naming
    log "Deploying job '$JOB_NAME' in region '$region'..."

    # Deploy the job with increased memory and environment variable for region
    gcloud run jobs deploy $JOB_NAME \
        --region=$region \
        --image=$FULL_IMAGE_PATH \
        --task-timeout=600 \
        --memory=1Gi \
        --cpu=1 \
        --command=k6 \
        --args="run,/home/k6/${TEST_SCRIPT}" \
        --set-env-vars="TARGET_URL=${TARGET_URL},REGION=${region},VUS=5,DURATION=60s" \
        --tasks=1 \
        --max-retries=1 \
        --quiet

    echo "$region,$JOB_NAME" >> $JOBS_FILE
    log "Job '$JOB_NAME' deployed successfully in region '$region'"
done

# 4. Execute All Jobs with Wait and Error Handling
log "Executing all jobs and waiting for completion..."

# Clear previous results
> cdn_performance_results.log
echo "=== CDN Performance Test Results - $(date) ===" >> cdn_performance_results.log

# Execute jobs sequentially and collect results
while IFS=, read -r region job_name; do
    log "Starting job '$job_name' in region '$region'..."
    
    # Execute the job with wait
    if gcloud run jobs execute $job_name --region=$region --wait --quiet; then
        result=0
        log "Job in $region completed successfully. Fetching logs..."
    else
        result=1
        log "Job in $region failed. Fetching logs for debugging..."
    fi
    
    # Add region header to results
    echo -e "\n===================================================" >> cdn_performance_results.log
    echo "REGION: $region" >> cdn_performance_results.log
    echo "JOB: $job_name" >> cdn_performance_results.log
    echo "===================================================" >> cdn_performance_results.log
    
    # Wait a moment for logs to be available
    sleep 5
    
    # Fetch logs with multiple attempts and better filtering
    log_content=""
    execution_name=""
    
    # First, get the execution name
    execution_name=$(gcloud run jobs executions list \
        --job=$job_name \
        --region=$region \
        --limit=1 \
        --format='value(metadata.name)' \
        --sort-by=~metadata.creationTimestamp 2>/dev/null || echo "")
    
    if [ -n "$execution_name" ]; then
        log "Found execution: $execution_name for job $job_name"
        
        # Try to get logs directly from the execution
        for attempt in 1 2 3; do
            log "Fetching logs for $region (attempt $attempt/3)..."
            
            # Method 1: Try to get logs from Cloud Logging with execution filter
            current_logs=$(gcloud logging read "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$job_name\" AND resource.labels.execution_name=\"$execution_name\"" \
                --format='value(textPayload)' \
                --project=$GCP_PROJECT_ID \
                --limit=2000 \
                --freshness=15m 2>/dev/null || echo "")
            
            # Method 2: If that fails, try without execution filter
            if [ -z "$current_logs" ]; then
                current_logs=$(gcloud logging read "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$job_name\"" \
                    --format='value(textPayload)' \
                    --project=$GCP_PROJECT_ID \
                    --limit=2000 \
                    --freshness=15m 2>/dev/null || echo "")
            fi
            
            # Method 3: Try to get logs from the execution description
            if [ -z "$current_logs" ]; then
                log "Trying to get logs from execution description..."
                current_logs=$(gcloud run jobs executions describe $execution_name \
                    --region=$region \
                    --format='value(status.logUri)' 2>/dev/null || echo "")
                    
                if [ -n "$current_logs" ]; then
                    current_logs="Execution completed. Log URI: $current_logs"
                fi
            fi
            
            if [ -n "$current_logs" ]; then
                log_content="$current_logs"
                break
            fi
            
            log "No logs found for $region in attempt $attempt, waiting..."
            sleep 10
        done
    else
        log "Could not find execution for job $job_name"
    fi
    
    if [ -n "$log_content" ]; then
        echo "$log_content" >> cdn_performance_results.log
        
        # Extract and display performance metrics immediately
        extract_metrics "$log_content" "$region"
    else
        log "WARNING: Could not retrieve logs for job $job_name in $region"
        echo "ERROR: Could not retrieve logs for this region" >> cdn_performance_results.log
        
        # Try to get job execution details for debugging
        latest_execution=$(gcloud run jobs executions list \
            --job=$job_name \
            --region=$region \
            --limit=1 \
            --format='value(metadata.name)' \
            --sort-by=~metadata.creationTimestamp 2>/dev/null || echo "")
        
        if [ -n "$latest_execution" ]; then
            log "Getting execution details for debugging: $latest_execution"
            gcloud run jobs executions describe $latest_execution --region=$region >> cdn_performance_results.log 2>&1 || true
        fi
    fi

    # Clean up job definition
    log "Cleaning up job definition '$job_name' in $region..."
    gcloud run jobs delete $job_name --region=$region --quiet || true
    
    log "Completed processing for region: $region"
    echo "---"
done < $JOBS_FILE

rm $JOBS_FILE

log "--- All Geo-Distributed Tests Complete ---"
log "Results have been saved to cdn_performance_results.log"

# Display summary
echo ""
echo "=== FINAL SUMMARY ==="
echo "Full results saved to: cdn_performance_results.log"
echo ""

# Show a quick summary of key metrics
log "Extracting key performance metrics..."
echo "=== KEY PERFORMANCE METRICS SUMMARY ===" >> cdn_performance_results.log
echo "" >> cdn_performance_results.log

for region in "${TEST_REGIONS[@]}"; do
    echo "Region: $region" >> cdn_performance_results.log
    grep -A 15 "CDN Performance Test Results for $region" cdn_performance_results.log | tail -15 >> cdn_performance_results.log || echo "No summary found for $region" >> cdn_performance_results.log
    echo "" >> cdn_performance_results.log
done

echo "=== Test completed! Check cdn_performance_results.log for detailed results ==="