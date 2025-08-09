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
# Add more regions as needed, e.g., "asia-east1", "europe-west1", etc.
TEST_REGIONS=(
  "asia-east2"
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

    # Deploy the job with increased memory and correct script path
    gcloud run jobs deploy $JOB_NAME \
        --region=$region \
        --image=$FULL_IMAGE_PATH \
        --task-timeout=600 \
        --memory=1Gi \
        --cpu=1 \
        --command=k6 \
        --args="run,/home/k6/${TEST_SCRIPT},-e,TARGET_URL=${TARGET_URL},--vus,5,--duration,30s" \
        --tasks=1 \
        --max-retries=1 \
        --quiet

    echo "$region,$JOB_NAME" >> $JOBS_FILE
    log "Job '$JOB_NAME' deployed successfully in region '$region'"
done

# 4. Execute All Jobs with Wait and Error Handling
log "Executing all jobs and waiting for completion..."

# Clear previous results
> all_results.log

while IFS=, read -r region job_name; do
    log "Executing and waiting for job '$job_name' in region '$region'..."
    
    # Execute the job with --wait to wait for completion
    if gcloud run jobs execute $job_name --region=$region --wait --quiet; then
        log "Job in $region completed successfully. Fetching logs..."
        echo -e "\n--- Results from Region: $region ---\n" >> all_results.log
        
        # Get the most recent execution for this job
        latest_execution=$(gcloud run jobs executions list \
            --job=$job_name \
            --region=$region \
            --limit=1 \
            --format='value(metadata.name)' \
            --sort-by=~metadata.creationTimestamp)
        
        if [ -n "$latest_execution" ]; then
            log "Found execution: $latest_execution"
            # Fetch logs for the specific execution
            gcloud logging read "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$job_name\"" \
                --format='value(textPayload)' \
                --project=$GCP_PROJECT_ID \
                --limit=1000 \
                --freshness=1d >> all_results.log
        else
            log "Warning: Could not find execution for job $job_name"
        fi
    else
        log "ERROR: Job execution failed in $region. Getting detailed error information..."
        
        # Get the failed execution details
        failed_execution=$(gcloud run jobs executions list \
            --job=$job_name \
            --region=$region \
            --limit=1 \
            --format='value(metadata.name)' \
            --sort-by=~metadata.creationTimestamp)
        
        if [ -n "$failed_execution" ]; then
            log "Failed execution ID: $failed_execution"
            log "Getting execution details for debugging..."
            gcloud run jobs executions describe $failed_execution --region=$region
            
            echo -e "\n--- ERROR from Region: $region ---\n" >> all_results.log
            echo "Execution failed: $failed_execution" >> all_results.log
            
            # Still try to get logs for debugging
            gcloud logging read "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$job_name\"" \
                --format='value(textPayload)' \
                --project=$GCP_PROJECT_ID \
                --limit=1000 \
                --freshness=1d >> all_results.log
        fi
    fi

    log "Cleaning up job definition '$job_name' in $region..."
    gcloud run jobs delete $job_name --region=$region --quiet
done < $JOBS_FILE

rm $JOBS_FILE

log "--- All Geo-Distributed Tests Complete ---"
log "Results have been saved to all_results.log"

# The script now outputs the raw log file. The master script can then call the report generator.
cat all_results.log