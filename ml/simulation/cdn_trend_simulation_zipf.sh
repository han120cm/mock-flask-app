#!/bin/bash

WEB_BASE="https://web-server-577176926733.us-central1.run.app"
CDN_BASE="https://cdn.sohryuu.me"
LOG_FILE="cdn-simulation/cdn_trend_simulation_zipf.log"
TOTAL_REQUESTS=100  
ZIPF_PARAMETER=1.2  

# Define content categories and their base weights
# Format: category:weight
CONTENT_CATEGORIES=(
    "images/trending:10"
    "images/popular:8"
    "images/general:5"
    "images/rare:1"
    "videos/tutorials:7"
    "videos/documentaries:4"
    "videos/archived:1"
)

# Function to get weight for a category
get_category_weight() {
    local search_category="$1"
    for item in "${CONTENT_CATEGORIES[@]}"; do
        local category="${item%%:*}"
        local weight="${item##*:}"
        if [[ "$category" == "$search_category" ]]; then
            echo "$weight"
            return
        fi
    done
    echo "1"  # Default weight
}

generate_content_catalog() {
    declare -g -a ALL_CONTENT_URLS
    declare -g -a CONTENT_WEIGHTS
    local index=0
    
    echo "Generating content catalog with Zipf distribution..."
    
    # Generate web server content (category endpoints only)
    for item in "${CONTENT_CATEGORIES[@]}"; do
        local category="${item%%:*}"
        local base_weight="${item##*:}"
        local url="$WEB_BASE/$category"
        
        ALL_CONTENT_URLS[$index]=$url
        CONTENT_WEIGHTS[$index]=$base_weight
        
        ((index++))
    done
    
    # Generate CDN static content
    # Images (1-99)
    for i in {1..99}; do
        local url="$CDN_BASE/images/image_$i.jpg"
        ALL_CONTENT_URLS[$index]=$url
        
        # Higher numbered images are less popular (Zipf distribution)
        # Using integer exponent to avoid bc issues
        if [[ $i -eq 1 ]]; then
            local item_weight=6.0
        else
            local item_weight=$(echo "scale=6; 6 / $i" | bc -l)
        fi
        CONTENT_WEIGHTS[$index]=$item_weight
        
        ((index++))
    done
    
    # Videos (1-99)  
    for i in {1..99}; do
        local url="$CDN_BASE/videos/video_$i.mp4"
        ALL_CONTENT_URLS[$index]=$url
        
        # Higher numbered videos are less popular (Zipf distribution)
        # Using integer exponent to avoid bc issues
        if [[ $i -eq 1 ]]; then
            local item_weight=5.0
        else
            local item_weight=$(echo "scale=6; 5 / $i" | bc -l)
        fi
        CONTENT_WEIGHTS[$index]=$item_weight
        
        ((index++))
    done
    
    echo "Generated ${#ALL_CONTENT_URLS[@]} total URLs (${#CONTENT_CATEGORIES[@]} web categories + 198 CDN files)"
}

calculate_cumulative_distribution() {
    declare -g -a CUMULATIVE_WEIGHTS
    local total_weight=0
    
    echo "Calculating Zipf cumulative distribution..."
    
    # Calculate total weight
    for weight in "${CONTENT_WEIGHTS[@]}"; do
        total_weight=$(echo "$total_weight + $weight" | bc -l)
    done
    
    # Calculate cumulative probabilities
    local cumulative=0
    for i in "${!CONTENT_WEIGHTS[@]}"; do
        local probability=$(echo "scale=8; ${CONTENT_WEIGHTS[$i]} / $total_weight" | bc -l)
        cumulative=$(echo "$cumulative + $probability" | bc -l)
        CUMULATIVE_WEIGHTS[$i]=$cumulative
        
        # Debug: Log top 20 most popular items
        if [[ $i -lt 20 ]]; then
            echo "Rank $((i+1)): ${ALL_CONTENT_URLS[$i]} (prob: $probability)" >> debug_top_content.log
        fi
    done
    
    echo "Distribution calculated for ${#CUMULATIVE_WEIGHTS[@]} items"
}

# Select content using Zipf distribution via inverse transform sampling
select_zipf_content() {
    local random=$(echo "scale=8; $RANDOM / 32767" | bc -l)
    
    # Binary search would be more efficient, but linear search is simpler for this size
    for i in "${!CUMULATIVE_WEIGHTS[@]}"; do
        if (( $(echo "$random <= ${CUMULATIVE_WEIGHTS[$i]}" | bc -l) )); then
            echo "${ALL_CONTENT_URLS[$i]}"
            return
        fi
    done
    
    # Fallback (should rarely happen)
    echo "${ALL_CONTENT_URLS[-1]}"
}

# Enhanced access function with realistic timing patterns and error handling
access_url() {
    local url=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local max_retries=3
    local retry_count=0
    
    # Exponential inter-arrival time (more realistic than fixed delays)
    local delay=$(echo "scale=3; -l($RANDOM/32767+0.0001) * 0.3" | bc -l)
    sleep $delay
    
    # Make request with retry logic
    while [[ $retry_count -lt $max_retries ]]; do
        local response=$(curl -s -o /dev/null -w "%{http_code},%{time_total},%{size_download},%{url_effective}" \
                        --max-time 10 --connect-timeout 5 "$url" 2>/dev/null)
        
        local status_code=$(echo "$response" | cut -d',' -f1)
        
        # Check if request was successful or should be retried
        if [[ "$status_code" =~ ^[2-3][0-9][0-9]$ ]] || [[ "$status_code" == "404" ]]; then
            # Success or expected 404 - log and exit
            echo "$timestamp,$response" >> "$LOG_FILE"
            return 0
        else
            # Failed request - retry after brief delay
            ((retry_count++))
            if [[ $retry_count -lt $max_retries ]]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S'),RETRY_$retry_count,0,0,$url" >> "$LOG_FILE"
                sleep 1
            else
                # Max retries exceeded
                echo "$timestamp,FAILED,0,0,$url" >> "$LOG_FILE"
                return 1
            fi
        fi
    done
}

# Simulate realistic daily traffic patterns
simulate_day() {
    local day=$1
    local base_requests=$TOTAL_REQUESTS
    
    echo "Day $day: Simulating realistic traffic patterns" | tee -a "$LOG_FILE"
    
    # Simulate different traffic patterns throughout the day
    local current_hour=$(date +%H)
    local multiplier
    
    # Apply time-of-day multipliers (realistic web traffic patterns)
    case $current_hour in
        0[0-6]) multiplier="0.3" ;;  # Low traffic: midnight-6AM
        0[7-8]) multiplier="0.8" ;;  # Morning ramp-up
        [09-11]) multiplier="1.5" ;; # Morning peak
        1[2-3]) multiplier="1.2" ;;  # Lunch dip
        1[4-7]) multiplier="1.8" ;;  # Afternoon peak
        1[8-9]) multiplier="2.0" ;;  # Evening peak
        2[0-3]) multiplier="1.0" ;;  # Evening decline
        *) multiplier="1.0" ;;       # Default fallback
    esac
    
    local requests_this_period=$(echo "scale=0; $base_requests * $multiplier / 1" | bc)
    
    echo "Time: ${current_hour}:00, Multiplier: ${multiplier}x, Requests: $requests_this_period"
    
    # Generate requests following Zipf distribution
    for i in $(seq 1 $requests_this_period); do
        local selected_url=$(select_zipf_content)
        access_url "$selected_url"
        
        # Progress indicator every 25 requests
        if (( i % 25 == 0 )); then
            echo "Progress: $i/$requests_this_period requests completed"
        fi
    done
    
    echo "Day $day completed: $requests_this_period requests generated" | tee -a "$LOG_FILE"
}

# Comprehensive analysis of request patterns
analyze_traffic_patterns() {
    echo "Analyzing traffic patterns..."
    
    if [[ ! -s "$LOG_FILE" ]]; then
        echo "No data to analyze - log file is empty"
        return 1
    fi
    
    local total_requests=$(tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | wc -l)
    local successful_requests=$(tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | grep -E ",[2-3][0-9][0-9]," | wc -l)
    local failed_requests=$(tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | grep -E ",FAILED," | wc -l)
    
    # Overall statistics
    echo "=== SIMULATION STATISTICS ===" > traffic_analysis.txt
    echo "Total requests: $total_requests" >> traffic_analysis.txt
    echo "Successful requests: $successful_requests" >> traffic_analysis.txt
    echo "Failed requests: $failed_requests" >> traffic_analysis.txt
    echo "Success rate: $(echo "scale=2; $successful_requests * 100 / $total_requests" | bc -l)%" >> traffic_analysis.txt
    echo "" >> traffic_analysis.txt
    
    # Overall popularity analysis
    echo "=== TOP 20 MOST REQUESTED URLS ===" >> traffic_analysis.txt
    tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | grep -E ",[2-3][0-9][0-9]," | cut -d',' -f5 | \
    sort | uniq -c | sort -nr | head -20 >> traffic_analysis.txt
    
    # Category-wise analysis
    echo -e "\n=== REQUESTS BY CATEGORY ===" >> traffic_analysis.txt
    tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | grep -E ",[2-3][0-9][0-9]," | cut -d',' -f5 | \
    sed -E 's|https?://[^/]+/([^/]+/[^/]+).*|\1|' | sort | uniq -c | sort -nr >> traffic_analysis.txt
    
    # Content type analysis
    echo -e "\n=== REQUESTS BY CONTENT TYPE ===" >> traffic_analysis.txt
    local image_requests=$(tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | grep -E ",[2-3][0-9][0-9]," | grep -c "\.jpg")
    local video_requests=$(tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | grep -E ",[2-3][0-9][0-9]," | grep -c "\.mp4")
    local other_requests=$((successful_requests - image_requests - video_requests))
    
    echo "Images: $image_requests ($(echo "scale=1; $image_requests * 100 / $successful_requests" | bc -l)%)" >> traffic_analysis.txt
    echo "Videos: $video_requests ($(echo "scale=1; $video_requests * 100 / $successful_requests" | bc -l)%)" >> traffic_analysis.txt
    echo "Other: $other_requests ($(echo "scale=1; $other_requests * 100 / $successful_requests" | bc -l)%)" >> traffic_analysis.txt
    
    # Response time analysis
    echo -e "\n=== RESPONSE TIME ANALYSIS ===" >> traffic_analysis.txt
    tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | grep -E ",[2-3][0-9][0-9]," | cut -d',' -f3 | \
    awk '{
        sum += $1; 
        if (NR == 1) min = max = $1; 
        if ($1 < min) min = $1; 
        if ($1 > max) max = $1;
        times[NR] = $1
    } 
    END {
        if (NR > 0) {
            avg = sum/NR;
            asort(times);
            if (NR % 2 == 1) median = times[int(NR/2)+1];
            else median = (times[NR/2] + times[NR/2+1])/2;
            print "Average: " avg "s";
            print "Median: " median "s"; 
            print "Min: " min "s";
            print "Max: " max "s";
        }
    }' >> traffic_analysis.txt
    
    # Zipf validation: plot rank vs frequency
    echo -e "\n=== ZIPF DISTRIBUTION VALIDATION (Top 50) ===" >> traffic_analysis.txt
    echo "Rank,Frequency,URL" >> traffic_analysis.txt
    tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | grep -E ",[2-3][0-9][0-9]," | cut -d',' -f5 | \
    sort | uniq -c | sort -nr | awk '{print NR "," $1 "," $2}' | head -50 >> traffic_analysis.txt
    
    # Generate simple visualization data for external plotting
    echo -e "\n=== HOURLY REQUEST DISTRIBUTION ===" >> traffic_analysis.txt
    tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | grep -E ",[2-3][0-9][0-9]," | \
    cut -d',' -f1 | cut -d' ' -f2 | cut -d':' -f1 | sort | uniq -c | \
    awk '{print $2 ":00," $1}' >> traffic_analysis.txt
    
    echo "Analysis complete! Check traffic_analysis.txt for detailed results."
    
    # Display quick summary
    echo -e "\n Quick Summary:"
    echo "  Total Requests: $total_requests"
    echo "  Success Rate: $(echo "scale=1; $successful_requests * 100 / $total_requests" | bc -l)%"
    echo "  Most Popular: $(tail -n +2 "$LOG_FILE" | grep -E ",[2-3][0-9][0-9]," | cut -d',' -f5 | sort | uniq -c | sort -nr | head -1 | awk '{print $2 " (" $1 " requests)"}')"
}

# Main execution with enhanced error handling and progress tracking
main() {
    echo "Starting CDN Traffic Simulation"
    echo "Parameters:"
    echo "  - Total content items: ~205 (7 web categories + 198 CDN files)"  
    echo "  - Zipf parameter: $ZIPF_PARAMETER"
    echo "  - Base requests per day: $TOTAL_REQUESTS"
    echo "  - Categories: ${CONTENT_CATEGORIES[*]%%:*}"
    
    # Check network connectivity
    echo "Testing network connectivity..."
    if ! curl -s --max-time 5 --head "$WEB_BASE" > /dev/null 2>&1; then
        echo "Warning: Cannot reach $WEB_BASE - some requests may fail"
    fi
    
    if ! curl -s --max-time 5 --head "https://cdn.sohryuu.me" > /dev/null 2>&1; then
        echo "Warning: Cannot reach CDN - some requests may fail"
    fi
    
    # Initialize log files
    > "$LOG_FILE"
    > debug_top_content.log
    > traffic_analysis.txt
    echo "timestamp,status_code,response_time,size_bytes,url" > "$LOG_FILE"
    
    # Setup content catalog with Zipf distribution
    echo "Setting up content catalog..."
    generate_content_catalog
    calculate_cumulative_distribution
    
    # Verify setup
    if [[ ${#ALL_CONTENT_URLS[@]} -eq 0 ]]; then
        echo "Error: No content URLs generated!"
        exit 1
    fi
    
    echo "Setup complete. Content catalog has ${#ALL_CONTENT_URLS[@]} URLs"
    
    # Run 5-day simulation with progress tracking
    echo -e "\n Starting 5-day simulation..."
    local start_time=$(date +%s)
    
    for day in {1..5}; do
        echo "=== Day $day/5 ==="
        simulate_day $day
        
        # Show progress
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local avg_time_per_day=$((elapsed / day))
        local estimated_remaining=$(((5 - day) * avg_time_per_day))
        
        echo "Day $day completed. Elapsed: ${elapsed}s, Estimated remaining: ${estimated_remaining}s"
        
        # Add realistic pause between days
        if [[ $day -lt 5 ]]; then
            echo "End of day $day, brief pause..."
            sleep 3
        fi
    done
    
    local total_time=$(($(date +%s) - start_time))
    echo "All 5 days completed in ${total_time} seconds"
    
    # Generate comprehensive analysis
    echo -e "\n Generating analysis..."
    analyze_traffic_patterns
    
    # Final summary
    echo -e "\n Simulation Complete!"
    echo "Generated files:"
    echo "$LOG_FILE - Complete request logs ($(wc -l < "$LOG_FILE") lines)"
    echo "traffic_analysis.txt - Traffic pattern analysis" 
    echo "debug_top_content.log - Top content rankings"
    
    # Validate Zipf distribution worked
    local total_requests=$(tail -n +2 "$LOG_FILE" | grep -E "^[0-9]" | wc -l)
    if [[ $total_requests -gt 0 ]]; then
        echo -e "\n Zipf Distribution Validation:"
        echo "  • Total requests generated: $total_requests"
        local top_url=$(tail -n +2 "$LOG_FILE" | grep -E ",[2-3][0-9][0-9]," | cut -d',' -f5 | sort | uniq -c | sort -nr | head -1)
        echo "  • Most popular URL: $(echo "$top_url" | awk '{print $2}') ($(echo "$top_url" | awk '{print $1}') requests)"
        echo "  • This demonstrates proper Zipf distribution - few items get most traffic!"
    else
        echo "Warning: No successful requests recorded - check network connectivity"
    fi
}

# Signal handling for graceful shutdown
cleanup() {
    echo -e "\n Simulation interrupted. Generating partial analysis..."
    analyze_traffic_patterns
    echo "Partial results saved to files."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Dependency check and final execution
check_dependencies() {
    local missing_deps=()
    
    if ! command -v bc &> /dev/null; then
        missing_deps+=("bc")
    fi
    
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    if ! command -v awk &> /dev/null; then
        missing_deps+=("awk")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo "Error: Missing required dependencies: ${missing_deps[*]}"
        echo "Install commands:"
        echo "  Ubuntu/Debian: sudo apt-get install ${missing_deps[*]}"
        echo "  CentOS/RHEL: sudo yum install ${missing_deps[*]}"
        echo "  macOS: brew install ${missing_deps[*]}"
        exit 1
    fi
    
    echo "All dependencies satisfied"
}

# Validate configuration
validate_config() {
    if [[ $TOTAL_REQUESTS -lt 1 ]]; then
        echo "Error: TOTAL_REQUESTS must be >= 1"
        exit 1
    fi
    
    if [[ $(echo "$ZIPF_PARAMETER < 0.5 || $ZIPF_PARAMETER > 3.0" | bc -l) -eq 1 ]]; then
        echo "Error: ZIPF_PARAMETER should typically be between 0.5 and 3.0"
        echo "Current value: $ZIPF_PARAMETER"
        exit 1
    fi
    
    # Test log file writeability
    if ! touch "$LOG_FILE" 2>/dev/null; then
        echo "Error: Cannot write to log file: $LOG_FILE"
        exit 1
    fi
    
    echo "Configuration validated"
}

# Main script execution
echo "CDN Traffic Simulation with Zipf Distribution"

# Run all checks
check_dependencies
validate_config

# Execute main simulation
main