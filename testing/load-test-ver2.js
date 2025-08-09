import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';

// Custom metrics for detailed CDN performance tracking
const imageLatency = new Trend('image_req_duration', true);
const videoLatency = new Trend('video_req_duration', true);
const cacheHitRate = new Rate('cache_hit_rate');
const videoCacheHitRate = new Rate('video_cache_hit_rate');
const imageCacheHitRate = new Rate('image_cache_hit_rate');
const videoRequestCount = new Counter('video_requests_total');
const imageRequestCount = new Counter('image_requests_total');

// --- Configuration ---
const APP_BASE_URL = __ENV.APP_URL || 'https://web-server-577176926733.us-central1.run.app';
const CDN_BASE_URL = __ENV.TARGET_URL || 'https://cdn.sohryuu.me';

const IMAGE_GROUPS = ['trending', 'popular', 'general', 'rare'];
const VIDEO_GROUPS = ['short-clips', 'documentaries', 'tutorials', 'archived'];

// Get current region for logging (if available)
const REGION = __ENV.REGION || 'unknown-region';

export const options = {
  vus: 5,
  duration: '30s',
  
  thresholds: {
    'http_req_duration': ['p(95)<5000'],
    'http_req_failed': ['rate<0.1'],
    'image_req_duration': ['p(95)<3000'],
    'video_req_duration': ['p(95)<10000'],
    'cache_hit_rate': ['rate>0.5'], // Expect at least 50% cache hit rate
  },
};

// Performance tracking object
let performanceStats = {
  region: REGION,
  videoStats: {
    requests: 0,
    totalLatency: 0,
    cacheHits: 0,
    errors: 0,
    minLatency: Infinity,
    maxLatency: 0
  },
  imageStats: {
    requests: 0,
    totalLatency: 0,
    cacheHits: 0,
    errors: 0,
    minLatency: Infinity,
    maxLatency: 0
  }
};

export default function() {
  // Mix of video and image browsing for comprehensive testing
  const testType = Math.random();
  
  if (testType < 0.6) {
    browseVideosWithMetrics();
  } else {
    browseImagesWithMetrics();
  }
  
  sleep(Math.random() * 2 + 1);
}

export function browseVideosWithMetrics() {
  console.log(`[${REGION}] Starting video browse test...`);
  
  // 1. Browse video category page first
  const randomVideoGroup = VIDEO_GROUPS[Math.floor(Math.random() * VIDEO_GROUPS.length)];
  let res = http.get(`${APP_BASE_URL}/videos/${randomVideoGroup}`, {
    timeout: '30s',
    tags: { 
      request_type: 'video_category', 
      category: randomVideoGroup,
      scenario: 'video_browse',
      source: 'app_server',
      region: REGION
    },
  });
  
  check(res, { 
    [`video category ${randomVideoGroup} loaded`]: (r) => r.status === 200,
  });
  
  sleep(1);

  // 2. Fetch actual video with detailed metrics
  const randomVideoIndex = Math.floor(Math.random() * 10) + 1; // 1-10 instead of 0-9
  const videoUrl = `${CDN_BASE_URL}/static/videos/video_${randomVideoIndex}.mp4`;
  
  console.log(`[${REGION}] Requesting video: ${videoUrl}`);
  
  const startTime = Date.now();
  res = http.get(videoUrl, {
    timeout: '60s', // Longer timeout for videos
    tags: {
      request_type: 'static_video',
      scenario: 'video_browse', 
      source: 'cdn',
      video_index: randomVideoIndex.toString(),
      region: REGION
    },
  });
  const endTime = Date.now();
  const requestDuration = endTime - startTime;
  
  // Update performance stats
  performanceStats.videoStats.requests++;
  videoRequestCount.add(1);
  
  if (res.status === 200) {
    // Record successful request metrics
    videoLatency.add(res.timings.duration);
    performanceStats.videoStats.totalLatency += res.timings.duration;
    performanceStats.videoStats.minLatency = Math.min(performanceStats.videoStats.minLatency, res.timings.duration);
    performanceStats.videoStats.maxLatency = Math.max(performanceStats.videoStats.maxLatency, res.timings.duration);
    
    // Check cache status
    const cacheStatus = res.headers['X-Cache-Status'] || res.headers['cf-cache-status'] || res.headers['x-cache'] || 'unknown';
    const isCacheHit = cacheStatus.toLowerCase().includes('hit') || cacheStatus.toLowerCase().includes('stale');
    
    cacheHitRate.add(isCacheHit);
    videoCacheHitRate.add(isCacheHit);
    
    if (isCacheHit) {
      performanceStats.videoStats.cacheHits++;
    }
    
    // Detailed logging
    const contentLength = res.headers['content-length'] || 'unknown';
    const serverHeader = res.headers['server'] || 'unknown';
    
    console.log(`[${REGION}] VIDEO SUCCESS - video_${randomVideoIndex}: Status=${res.status} | Duration=${res.timings.duration.toFixed(2)}ms | Cache=${cacheStatus} | Size=${contentLength} bytes | Server=${serverHeader}`);
    
  } else {
    performanceStats.videoStats.errors++;
    console.log(`[${REGION}] VIDEO ERROR - video_${randomVideoIndex}: Status=${res.status} | Duration=${requestDuration}ms | Error=${res.error || 'Unknown'}`);
  }
  
  const videoChecks = check(res, { 
    'video request successful': (r) => r.status === 200,
    'video response time reasonable': (r) => r.timings.duration < 15000,
  });
  
  if (!videoChecks) {
    console.error(`[${REGION}] Video request checks failed for ${videoUrl}`);
  }
}

export function browseImagesWithMetrics() {
  console.log(`[${REGION}] Starting image browse test...`);
  
  // 1. Browse main page
  let res = http.get(`${APP_BASE_URL}/`, {
    timeout: '30s',
    tags: { 
      request_type: 'homepage',
      scenario: 'image_browse',
      source: 'app_server',
      region: REGION
    },
  });
  
  check(res, { 
    'home page loaded': (r) => r.status === 200,
  });
  
  sleep(1);

  // 2. Browse image category
  const randomImageGroup = IMAGE_GROUPS[Math.floor(Math.random() * IMAGE_GROUPS.length)];
  res = http.get(`${CDN_BASE_URL}/images/${randomImageGroup}`, {
    timeout: '30s',
    tags: { 
      request_type: 'image_category', 
      category: randomImageGroup,
      scenario: 'image_browse',
      source: 'cdn',
      region: REGION
    },
  });
  
  check(res, { 
    [`image category ${randomImageGroup} loaded`]: (r) => r.status === 200,
  });
  
  sleep(1);

  // 3. Fetch actual image with detailed metrics
  const randomImageIndex = Math.floor(Math.random() * 10) + 1; // 1-10 instead of 0-9
  const imageUrl = `${CDN_BASE_URL}/static/images/image_${randomImageIndex}.jpg`;
  
  console.log(`[${REGION}] Requesting image: ${imageUrl}`);
  
  const startTime = Date.now();
  res = http.get(imageUrl, {
    timeout: '30s',
    tags: { 
      request_type: 'static_image',
      scenario: 'image_browse',
      source: 'cdn',
      image_index: randomImageIndex.toString(),
      region: REGION
    },
  });
  const endTime = Date.now();
  const requestDuration = endTime - startTime;
  
  // Update performance stats
  performanceStats.imageStats.requests++;
  imageRequestCount.add(1);
  
  if (res.status === 200) {
    // Record successful request metrics
    imageLatency.add(res.timings.duration);
    performanceStats.imageStats.totalLatency += res.timings.duration;
    performanceStats.imageStats.minLatency = Math.min(performanceStats.imageStats.minLatency, res.timings.duration);
    performanceStats.imageStats.maxLatency = Math.max(performanceStats.imageStats.maxLatency, res.timings.duration);
    
    // Check cache status
    const cacheStatus = res.headers['X-Cache-Status'] || res.headers['cf-cache-status'] || res.headers['x-cache'] || 'unknown';
    const isCacheHit = cacheStatus.toLowerCase().includes('hit') || cacheStatus.toLowerCase().includes('stale');
    
    cacheHitRate.add(isCacheHit);
    imageCacheHitRate.add(isCacheHit);
    
    if (isCacheHit) {
      performanceStats.imageStats.cacheHits++;
    }
    
    // Detailed logging
    const contentLength = res.headers['content-length'] || 'unknown';
    const serverHeader = res.headers['server'] || 'unknown';
    
    console.log(`[${REGION}] IMAGE SUCCESS - image_${randomImageIndex}: Status=${res.status} | Duration=${res.timings.duration.toFixed(2)}ms | Cache=${cacheStatus} | Size=${contentLength} bytes | Server=${serverHeader}`);
    
  } else {
    performanceStats.imageStats.errors++;
    console.log(`[${REGION}] IMAGE ERROR - image_${randomImageIndex}: Status=${res.status} | Duration=${requestDuration}ms | Error=${res.error || 'Unknown'}`);
  }
  
  check(res, { 
    'static image request completed': (r) => r.status === 200,
    'image response time reasonable': (r) => r.timings.duration < 5000,
  });
}

export function setup() {
  console.log(`=== CDN Performance Test Starting ===`);
  console.log(`Region: ${REGION}`);
  console.log(`Target app: ${APP_BASE_URL}`);
  console.log(`CDN: ${CDN_BASE_URL}`);
  console.log(`Test duration: ${options.duration}`);
  console.log(`Virtual users: ${options.vus}`);
  console.log(`=====================================`);
}

export function teardown(data) {
  console.log(`\n=== CDN Performance Test Results for ${REGION} ===`);
  
  // Calculate averages
  const videoAvgLatency = performanceStats.videoStats.requests > 0 ? 
    (performanceStats.videoStats.totalLatency / performanceStats.videoStats.requests).toFixed(2) : 'N/A';
  const imageAvgLatency = performanceStats.imageStats.requests > 0 ? 
    (performanceStats.imageStats.totalLatency / performanceStats.imageStats.requests).toFixed(2) : 'N/A';
  
  const videoCacheHitPercent = performanceStats.videoStats.requests > 0 ?
    ((performanceStats.videoStats.cacheHits / performanceStats.videoStats.requests) * 100).toFixed(1) : 'N/A';
  const imageCacheHitPercent = performanceStats.imageStats.requests > 0 ?
    ((performanceStats.imageStats.cacheHits / performanceStats.imageStats.requests) * 100).toFixed(1) : 'N/A';

  console.log(`\n📹 VIDEO PERFORMANCE:`);
  console.log(`   Requests: ${performanceStats.videoStats.requests}`);
  console.log(`   Average Latency: ${videoAvgLatency}ms`);
  console.log(`   Min Latency: ${performanceStats.videoStats.minLatency === Infinity ? 'N/A' : performanceStats.videoStats.minLatency.toFixed(2) + 'ms'}`);
  console.log(`   Max Latency: ${performanceStats.videoStats.maxLatency || 'N/A'}ms`);
  console.log(`   Cache Hit Rate: ${videoCacheHitPercent}%`);
  console.log(`   Errors: ${performanceStats.videoStats.errors}`);

  console.log(`\n🖼️  IMAGE PERFORMANCE:`);
  console.log(`   Requests: ${performanceStats.imageStats.requests}`);
  console.log(`   Average Latency: ${imageAvgLatency}ms`);
  console.log(`   Min Latency: ${performanceStats.imageStats.minLatency === Infinity ? 'N/A' : performanceStats.imageStats.minLatency.toFixed(2) + 'ms'}`);
  console.log(`   Max Latency: ${performanceStats.imageStats.maxLatency || 'N/A'}ms`);
  console.log(`   Cache Hit Rate: ${imageCacheHitPercent}%`);
  console.log(`   Errors: ${performanceStats.imageStats.errors}`);

  console.log(`\n================================================`);
}