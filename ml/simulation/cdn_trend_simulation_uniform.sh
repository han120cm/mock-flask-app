#!/bin/bash

WEB_BASE="origin-server.com"
CDN_BASE="cdn-server.com"
LOG_FILE="cdn_trend_simulation.log"
REPEAT_BASE=3 

access_url() {
  local url=$1
  local repeat=$2
  for i in $(seq 1 $repeat); do
    curl -s -o /dev/null -w "%{http_code} %{time_total} %{url_effective}\n" "$url" >> "$LOG_FILE"
    sleep 0.1
  done
}

simulate_day() {
  local day=$1
  echo "Simulating Day $day: $(date)" | tee -a "$LOG_FILE"

  case $day in
    1)
      access_url "$WEB_BASE/images/trending" $((REPEAT_BASE*2))
      access_url "$WEB_BASE/images/general"  $REPEAT_BASE
      access_url "$WEB_BASE/videos/tutorials" $REPEAT_BASE
      access_url "$CDN_BASE/images/image_1.jpg" $REPEAT_BASE
      access_url "$CDN_BASE/videos/video_2.mp4" $REPEAT_BASE
      ;;
    2)
      access_url "$WEB_BASE/images/trending" $((REPEAT_BASE*3))
      access_url "$WEB_BASE/images/rare" 1
      access_url "$WEB_BASE/videos/archived" 1
      access_url "$CDN_BASE/images/image_2.jpg" $((REPEAT_BASE*2))
      access_url "$CDN_BASE/videos/video_4.mp4" $REPEAT_BASE
      ;;
    3)
      access_url "$WEB_BASE/images/popular" $((REPEAT_BASE*5))
      access_url "$WEB_BASE/images/trending" $((REPEAT_BASE*3))
      access_url "$CDN_BASE/images/image_3.jpg" $((REPEAT_BASE*4))
      access_url "$CDN_BASE/videos/video_7.mp4" $REPEAT_BASE
      ;;
    4)
      access_url "$WEB_BASE/videos/documentaries" $((REPEAT_BASE*2))
      access_url "$WEB_BASE/videos/tutorials" $((REPEAT_BASE*2))
      access_url "$CDN_BASE/videos/video_10.mp4" $((REPEAT_BASE*2))
      ;;
    5)
      access_url "$WEB_BASE/images/rare" 1
      access_url "$WEB_BASE/videos/documentaries" $((REPEAT_BASE*6))
      access_url "$CDN_BASE/videos/video_5.mp4" $((REPEAT_BASE*5))
      access_url "$CDN_BASE/videos/video_6.mp4" $((REPEAT_BASE*4))
      ;;
    *)
      echo "Unknown day $day"
      ;;
  esac

  echo "Done Day $day" | tee -a "$LOG_FILE"
  sleep 1
}

for day in {1..5}; do
  simulate_day $day
done
