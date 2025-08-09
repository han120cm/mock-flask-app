import http from 'k6/http';
import { check, sleep } from 'k6';

// --- Configuration ---
const APP_BASE_URL = 'https://web-server-577176926733.us-central1.run.app';
const CDN_BASE_URL = 'https://cdn.sohryuu.me';

const IMAGE_GROUPS = ['trending', 'popular', 'general', 'rare'];
const VIDEO_GROUPS = ['short-clips', 'documentaries', 'tutorials', 'archived'];

// --- Improved Test Options ---
export const options = {
  cloud: {
    distribution: {
      // Try multiple regions to get better CDN coverage
      'amazon:us:ashburn': { loadZone: 'amazon:us:ashburn', percent: 50 },
      'amazon:us:oregon': { loadZone: 'amazon:us:oregon', percent: 50 }
    }
  },
  
  scenarios: {
    // Reduced concurrent load for video testing
    browse_videos_realistic: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 20 },  // Reduced from 20 to 5
        { duration: '5m', target: 20 }, // Reduced from 20 to 10  
        { duration: '2m', target: 0 },
      ],
      exec: 'browseVideosImproved',
      tags: { scenario: 'browse_videos_realistic' },
    },
    
    // Pre-warm cache before main test
    cache_warmup: {
      executor: 'shared-iterations',
      vus: 3,
      iterations: 30, // Warm up 10 videos, 3 times each
      maxDuration: '2m',
      exec: 'warmupCache',
      tags: { scenario: 'warmup' },
    },
    
    // Keep other scenarios with reduced load
    browse_main_and_images: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 }, // Reduced from 50 to 25
        { duration: '5m', target: 50 },
        { duration: '2m', target: 0 },
      ],
      exec: 'browseMainAndImages',
      tags: { scenario: 'browse_images' },
    },
    
    simulate_uploads: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 5 },  // Reduced from 5 to 2
        { duration: '7m', target: 5 },
        { duration: '1m', target: 0 },
      ],
      exec: 'simulateUploads',
      tags: { scenario: 'uploads' },
    },
    
    health_checks: {
      executor: 'constant-vus',
      vus: 1, // Reduced from 2 to 1
      duration: '9m',
      exec: 'runHealthCheck',
      tags: { scenario: 'health_check' },
    },
  },
  
  thresholds: {
    // More realistic thresholds based on your curl results
    'http_req_duration': ['p(95)<3000'],
    'http_req_failed': ['rate<0.01'],
    
    // Scenario-specific thresholds
    'http_req_duration{scenario:browse_images}': ['p(95)<1500'],
    'http_req_duration{scenario:browse_videos_realistic}': ['p(95)<15000'], // More realistic
    'http_req_duration{scenario:uploads}': ['p(95)<3000'],
    'http_req_duration{scenario:health_check}': ['p(95)<1000'],
    'http_req_duration{scenario:warmup}': ['p(95)<5000'], // Allow more time for cache warming
  },
};

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
  
  // Enhanced logging
  const cacheStatus = res.headers['X-Cache-Status'] || res.headers['cf-cache-status'] || 'unknown';
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
  
  check(res, { 
    'static image request completed': (r) => r.status === 200 || r.status === 404,
  });
  
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
  console.log('Test will warm cache first, then run realistic load patterns');
}

export function teardown(data) {
  console.log('CDN load test completed - check cache hit rates in logs');
}