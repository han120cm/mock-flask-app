import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate } from 'k6/metrics';

// Custom Trend metrics to track latency for specific content types
const imageLatency = new Trend('image_req_duration', true);
const videoLatency = new Trend('video_req_duration', true);
const cacheHitRate = new Rate('cache_hit_rate');

// --- Configuration ---
// Use environment variables for URLs, with fallbacks for local testing
const APP_BASE_URL = __ENV.APP_URL || 'https://web-server-577176926733.us-central1.run.app';
const CDN_BASE_URL = __ENV.TARGET_URL || 'https://cdn.sohryuu.me';

const IMAGE_GROUPS = ['trending', 'popular', 'general', 'rare'];
const VIDEO_GROUPS = ['short-clips', 'documentaries', 'tutorials', 'archived'];

// --- Simplified Options for Cloud Run ---
export const options = {
  // Simplified for single region testing
  vus: 5, // Can be overridden by CLI
  duration: '30s', // Can be overridden by CLI
  
  thresholds: {
    'http_req_duration': ['p(95)<5000'],
    'http_req_failed': ['rate<0.1'],
  },
};

// ADD THIS: Default function that k6 expects
export default function() {
  // Simple test that calls one of your existing functions
  browseMainAndImages();
}

// Cache warmup function - runs first to pre-load videos
export function warmupCache() {
  const videoIndex = Math.floor(Math.random() * 10);
  const videoUrl = `${CDN_BASE_URL}/static/videos/video_${videoIndex}.mp4`;
  
  const res = http.get(videoUrl, {
    timeout: '60s', // Longer timeout for initial cache fill
    tags: {
      request_type: 'cache_warmup',
      scenario: 'warmup',
      source: 'cdn',
      video_index: videoIndex.toString()
    },
  });
  
  const cacheStatus = res.headers['X-Cache-Status'] || res.headers['cf-cache-status'] || 'unknown';
  console.log(`Warmup video_${videoIndex}: ${res.status} in ${res.timings.duration}ms, Cache: ${cacheStatus}`);
  
  check(res, { 
    'warmup request completed': (r) => r.status === 200 || r.status === 404,
  });
  
  sleep(1);
}

// Improved video browsing with better diagnostics
export function browseVideosImproved() {
  // 1. Browse video category page
  const randomVideoGroup = VIDEO_GROUPS[Math.floor(Math.random() * VIDEO_GROUPS.length)];
  let res = http.get(`${APP_BASE_URL}/videos/${randomVideoGroup}`, {
    timeout: '30s',
    tags: { 
      request_type: 'video_category', 
      category: randomVideoGroup,
      scenario: 'browse_videos_realistic',
      source: 'app_server'
    },
  });
  
  check(res, { 
    [`video category ${randomVideoGroup} loaded`]: (r) => r.status === 200,
    'video category response time OK': (r) => r.timings.duration < 3000,
  });
  
  sleep(Math.random() * 2 + 1);

  // 2. Fetch video with improved error handling and diagnostics
  const randomVideoIndex = Math.floor(Math.random() * 10);
  const videoUrl = `${CDN_BASE_URL}/static/videos/video_${randomVideoIndex}.mp4`;
  
  const startTime = Date.now();
  res = http.get(videoUrl, {
    timeout: '60s',
    tags: {
      request_type: 'static_video',
      scenario: 'browse_videos_realistic', 
      source: 'cdn',
      video_index: randomVideoIndex.toString()
    },
  });
  const endTime = Date.now();
  videoLatency.add(res.timings.duration); // Add video latency to our custom trend
  
  // Enhanced logging
  const cacheStatus = res.headers['X-Cache-Status'] || res.headers['cf-cache-status'] || 'unknown';
  cacheHitRate.add(cacheStatus.toLowerCase().includes('hit')); // Add true for HIT, false otherwise
  const contentLength = res.headers['content-length'] || 'unknown';
  const serverHeader = res.headers['server'] || 'unknown';
  
  console.log(`Video ${randomVideoIndex}: ${res.status} | ${endTime - startTime}ms | Cache: ${cacheStatus} | Size: ${contentLength} | Server: ${serverHeader}`);
  
  const videoChecks = check(res, { 
    'video request successful': (r) => r.status === 200,
    // 'video response time reasonable': (r) => r.timings.duration < 15000, // 10s max
    'video served from cache': (r) => {
      const cache = r.headers['X-Cache-Status'] || r.headers['cf-cache-status'] || '';
      return cache.toLowerCase().includes('hit') || cache.toLowerCase().includes('stale');
    },
  });
  
  if (!videoChecks) {
    console.error(`Video request failed: Status=${res.status}, Duration=${res.timings.duration}ms, URL=${videoUrl}`);
  }
  
  sleep(Math.random() * 3 + 2);
}

// Keep your existing functions but with improved error handling
export function browseMainAndImages() {
  let res = http.get(`${APP_BASE_URL}/`, {
    timeout: '30s',
    tags: { 
      request_type: 'homepage',
      scenario: 'browse_images',
      source: 'app_server'
    },
  });
  
  check(res, { 
    'home page loaded': (r) => r.status === 200,
    'home page response time OK': (r) => r.timings.duration < 3000,
  });
  
  sleep(Math.random() * 2 + 1);

  const randomImageGroup = IMAGE_GROUPS[Math.floor(Math.random() * IMAGE_GROUPS.length)];
  res = http.get(`${CDN_BASE_URL}/images/${randomImageGroup}`, {
    timeout: '30s',
    tags: { 
      request_type: 'image_category', 
      category: randomImageGroup,
      scenario: 'browse_images',
      source: 'cdn'
    },
  });
  
  check(res, { 
    [`image category ${randomImageGroup} loaded`]: (r) => r.status === 200,
    'image category response time OK': (r) => r.timings.duration < 2000,
  });
  
  sleep(Math.random() * 2 + 1);

  const randomImageIndex = Math.floor(Math.random() * 10);
  res = http.get(`${CDN_BASE_URL}/static/images/image_${randomImageIndex}.jpg`, {
    timeout: '30s',
    tags: { 
      request_type: 'static_image',
      scenario: 'browse_images',
      source: 'cdn',
      image_index: randomImageIndex.toString()
    },
  });

  const imageCacheStatus = res.headers['X-Cache-Status'] || res.headers['cf-cache-status'] || 'unknown';
  cacheHitRate.add(imageCacheStatus.toLowerCase().includes('hit'));
  
  check(res, { 
    'static image request completed': (r) => r.status === 200 || r.status === 404,
  });
  imageLatency.add(res.timings.duration); // Add image latency to our custom trend
  
  sleep(Math.random() * 1 + 0.5);
}

export function simulateUploads() {
  let res = http.get(`${APP_BASE_URL}/upload`, {
    timeout: '30s',
    tags: { 
      request_type: 'upload_page',
      scenario: 'uploads',
      source: 'app_server'
    },
  });
  
  check(res, { 
    'upload page loaded': (r) => r.status === 200,
    'upload page response time OK': (r) => r.timings.duration < 3000,
  });
  
  sleep(Math.random() * 5 + 3);
}

export function runHealthCheck() {
  const res = http.get(`${APP_BASE_URL}/health`, {
    timeout: '15s',
    tags: { 
      request_type: 'health_check',
      scenario: 'health_check',
      source: 'app_server'
    },
  });
  
  check(res, { 
    'health check ok': (r) => r.status === 200,
    'health check response time OK': (r) => r.timings.duration < 1000,
  });
  
  sleep(10);
}

export function setup() {
  console.log('Starting improved CDN load test');
  console.log(`Target app: ${APP_BASE_URL}`);
  console.log(`CDN: ${CDN_BASE_URL}`);
  console.log('Test will run a simplified version for geo-distributed testing');
}

export function teardown(data) {
  console.log('CDN load test completed - check cache hit rates in logs');
}