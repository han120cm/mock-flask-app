import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// --- Custom Metrics for Grafana Validation ---
const throughputCounter = new Counter('custom_throughput');
const latencyTrend = new Trend('custom_latency');
const cacheHitRate = new Rate('cache_hit_rate');
const availabilityRate = new Rate('availability_rate');

// --- Configuration ---
// IMPORTANT: Replace with your actual Cloud Run URL and CDN URL
const APP_BASE_URL = 'https://web-server-577176926733.us-central1.run.app';
const CDN_BASE_URL = 'https://cdn.sohryuu.me';

// List of image and video groups to simulate browsing
const IMAGE_GROUPS = ['trending', 'popular', 'general', 'rare'];
const VIDEO_GROUPS = ['short-clips', 'documentaries', 'tutorials', 'archived'];

// --- Test Options with Geographic Distribution and Scenarios ---
export const options = {
  // K6 Cloud configuration for Free tier (single load zone)
  cloud: {
    distribution: {
      // Using US East as primary load zone (100% of traffic)
      // For multi-region, you'd configure percentages for other zones here
      'amazon:us:ashburn': { loadZone: 'amazon:us:ashburn', percent: 100 }
    }
  },
  
  scenarios: {
    // Scenario 1: Simulate users browsing main pages and image categories via CDN
    browse_main_and_images: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 20 }, // Ramp up to 50 VUs over 2 minutes
        { duration: '5m', target: 20 }, // Stay at 50 VUs for 5 minutes
        { duration: '2m', target: 0 },  // Ramp down to 0 VUs over 2 minutes
      ],
      exec: 'browseMainAndImages',
      tags: { scenario: 'browse_images' },
    },
    
    // Scenario 2: Simulate users browsing video categories via CDN
    browse_videos: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 20 }, // Ramp up to 20 VUs over 2 minutes
        { duration: '5m', target: 20 }, // Stay at 20 VUs for 5 minutes
        { duration: '2m', target: 0 },  // Ramp down to 0 VUs over 2 minutes
      ],
      exec: 'browseVideos',
      tags: { scenario: 'browse_videos' },
    },
    
    // Scenario 3: Simulate occasional content uploads (hits the app origin directly)
    simulate_uploads: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 5 },  // Ramp up to 5 VUs over 1 minute
        { duration: '7m', target: 5 },  // Stay at 5 VUs for 7 minutes
        { duration: '1m', target: 0 },  // Ramp down to 0 VUs over 1 minute
      ],
      exec: 'simulateUploads',
      tags: { scenario: 'uploads' },
    },
    
    // Scenario 4: Simulate health checks (hits the app origin directly)
    health_checks: {
      executor: 'constant-vus',
      vus: 2,
      duration: '9m', // Total test duration
      exec: 'runHealthCheck',
      tags: { scenario: 'health_check' },
    },
  },
  
  thresholds: {
    // Global thresholds for all scenarios
    'http_req_duration': ['p(95)<1000'], // 95% of requests should be below 2000ms (adjusted for global latency)
    'http_req_failed': ['rate<0.01'],    // Less than 1% of requests should fail
    
    // Scenario-specific thresholds
    'http_req_duration{scenario:browse_images}': ['p(95)<1000'],
    'http_req_duration{scenario:browse_videos}': ['p(95)<1000'],
    'http_req_duration{scenario:uploads}': ['p(95)<2000'],
    'http_req_duration{scenario:health_check}': ['p(95)<1000'],

    // Custom metric thresholds for Grafana validation
    'custom_throughput': ['count>0'], // Ensure some requests are made
    'custom_latency': ['p(50)<1000', 'p(95)<3000'], // Median < 1s, 95% < 3s
    'cache_hit_rate': ['rate>=0'], // Just ensure it's being reported
    'availability_rate': ['rate>0.99'], // Availability > 99%
  },
};

// --- Functions for each Scenario ---

// Simulates browsing the home page and then random image categories
export function browseMainAndImages() {
  // 1. Visit the main home page (served by Cloud Run)
  let res = http.get(`${APP_BASE_URL}/`, {
    timeout: '30s',
    tags: { 
      request_type: 'homepage',
      scenario: 'browse_images',
      source: 'app_server'
    },
  });
  
  const homepageCheck = check(res, { 
    'home page loaded (app)': (r) => r.status === 200,
    'home page response time OK': (r) => r.timings.duration < 3000,
  });
  
  if (!homepageCheck) {
    console.error(`Homepage request failed: ${res.status}`);
  }
  
  // Record custom metrics
  throughputCounter.add(1);
  latencyTrend.add(res.timings.duration);
  availabilityRate.add(res.status >= 200 && res.status < 400);
  cacheHitRate.add(isCacheHit(res));

  sleep(Math.random() * 2 + 1);

  // 2. Browse a random image category (served by CDN)
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
    [`image category ${randomImageGroup} loaded (cdn)`]: (r) => r.status === 200,
    'image category response time OK': (r) => r.timings.duration < 2000,
  });
  
  // Record custom metrics
  throughputCounter.add(1);
  latencyTrend.add(res.timings.duration);
  availabilityRate.add(res.status >= 200 && res.status < 400);
  cacheHitRate.add(isCacheHit(res));

  sleep(Math.random() * 2 + 1);

  // 3. Fetch a specific static image from CDN
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
  
  // Even if request fails (404), we still want to measure response time
  check(res, { 
    'static image request completed': (r) => r.status === 200 || r.status === 404,
    'static image response time measured': (r) => r.timings.duration > 0,
  });
  
  // Record custom metrics
  throughputCounter.add(1);
  latencyTrend.add(res.timings.duration);
  availabilityRate.add(res.status >= 200 && res.status < 500); // 404 is acceptable for static images
  cacheHitRate.add(isCacheHit(res));

  sleep(Math.random() * 1 + 0.5);
}

// Simulates browsing random video categories
export function browseVideos() {
  // Browse a random video category (served by CDN)
  const randomVideoGroup = VIDEO_GROUPS[Math.floor(Math.random() * VIDEO_GROUPS.length)];
  const res = http.get(`${CDN_BASE_URL}/videos/${randomVideoGroup}`, {
    timeout: '30s',
    tags: { 
      request_type: 'video_category', 
      category: randomVideoGroup,
      scenario: 'browse_videos',
      source: 'cdn'
    },
  });
  
  check(res, {
    [`video category ${randomVideoGroup} loaded (cdn)`]: (r) => r.status === 200,
    'video category response time OK': (r) => r.timings.duration < 2500,
    'video category response time measured': (r) => r.timings.duration > 0,
  });
  
  // Record custom metrics
  throughputCounter.add(1);
  latencyTrend.add(res.timings.duration);
  availabilityRate.add(res.status >= 200 && res.status < 400);
  cacheHitRate.add(isCacheHit(res));

  sleep(Math.random() * 3 + 2);
}

// Simulates an upload process (simplified: just hitting the upload page)
export function simulateUploads() {
  // Go to the upload page (served by Cloud Run)
  let res = http.get(`${APP_BASE_URL}/upload`, {
    timeout: '30s',
    tags: { 
      request_type: 'upload_page',
      scenario: 'uploads',
      source: 'app_server'
    },
  });
  
  const uploadCheck = check(res, { 
    'upload page loaded (app)': (r) => r.status === 200,
    'upload page response time OK': (r) => r.timings.duration < 3000,
    'upload page response time measured': (r) => r.timings.duration > 0,
  });
  
  if (!uploadCheck) {
    console.error(`Upload page request failed: ${res.status}`);
  }
  
  // Record custom metrics
  throughputCounter.add(1);
  latencyTrend.add(res.timings.duration);
  availabilityRate.add(res.status >= 200 && res.status < 400);
  // Cache hit is not relevant for app server requests, so we don't add it here

  sleep(Math.random() * 5 + 3);
}

// Simulates a health check request
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
    'health check response time measured': (r) => r.timings.duration > 0,
  });
  
  // Record custom metrics
  throughputCounter.add(1);
  latencyTrend.add(res.timings.duration);
  availabilityRate.add(res.status >= 200 && res.status < 400);
  // Cache hit is not relevant for app server requests

  sleep(10);
}

// Function to detect cache hit based on response headers
function isCacheHit(response) {
  const cacheHeaders = [
    'x-cache-status',
    'x-cache',
    'cf-cache-status', // Cloudflare
    'x-served-by',
    'age'
  ];
  
  // Check common cache indicators
  for (let header of cacheHeaders) {
    const value = response.headers[header];
    if (value) {
      // Common cache hit indicators
      if (value.toLowerCase().includes('hit') || 
          value.toLowerCase().includes('cached') ||
          parseInt(value) > 0) { // Age header > 0 usually means cached
        return true;
      }
    }
  }
  
  // Additional logic: if response time is very fast, likely from cache
  return response.timings.duration < 250; // < 50ms likely cached
}

// Setup function (runs once per VU at the start)
export function setup() {
  console.log('=== K6 Grafana Metrics Validation Test ===');
  console.log('Target app:', APP_BASE_URL);
  console.log('CDN:', CDN_BASE_URL);
  console.log('Image Categories:', IMAGE_GROUPS);
  console.log('Video Categories:', VIDEO_GROUPS);
  console.log('Expected to validate: Throughput, Latency, Cache Hit Rate, Availability');
  
  // Test connectivity to both endpoints
  console.log('Testing connectivity...');
  
  try {
    const appTest = http.get(`${APP_BASE_URL}/health`, { timeout: '10s' });
    console.log(`App server connectivity: ${appTest.status} (${appTest.timings.duration}ms)`);
  } catch (e) {
    console.error(`App server connectivity failed: ${e}`);
  }
  
  try {
    const cdnTest = http.get(`${CDN_BASE_URL}/`, { timeout: '10s' });
    console.log(`CDN connectivity: ${cdnTest.status} (${cdnTest.timings.duration}ms)`);
  } catch (e) {
    console.error(`CDN connectivity failed: ${e}`);
  }
  
  return { startTime: Date.now() };
}

// Teardown function (runs once at the end of the test)
export function teardown(data) {
  const testDuration = (Date.now() - data.startTime) / 1000;
  console.log(`\n=== Test Completed in ${testDuration}s ===`);
  console.log('Check your Grafana dashboard to compare:');
  console.log('1. Throughput: Should show requests/sec matching K6 rate');
  console.log('2. Latency: Compare p50, p95 with Grafana metrics');
  console.log('3. Cache Hit %: Compare with cache_hit_rate metric');
  console.log('4. Availability: Should be >99% if servers are healthy');
}

// Custom summary for detailed comparison
export function handleSummary(data) {
  const summary = {
    'stdout': `
=== GRAFANA VALIDATION SUMMARY ===

THROUGHPUT COMPARISON:
- K6 Total Requests: ${data.metrics.http_reqs.values.count}
- K6 Requests/sec: ${(data.metrics.http_reqs.values.count / (data.state.testRunDurationMs / 1000)).toFixed(2)}
- Duration: ${(data.state.testRunDurationMs / 1000).toFixed(2)}s

LATENCY COMPARISON:
- K6 Average: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
- K6 Median (p50): ${data.metrics.http_req_duration.values['p(50)'].toFixed(2)}ms
- K6 95th percentile: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms

AVAILABILITY:
- K6 Success Rate: ${((1 - data.metrics.http_req_failed.values.rate) * 100).toFixed(2)}%
- K6 Failed Requests: ${data.metrics.http_req_failed.values.count}

CACHE PERFORMANCE:
- K6 Cache Hit Rate: ${(data.metrics.cache_hit_rate.values.rate * 100).toFixed(2)}%

INSTRUCTIONS FOR GRAFANA COMPARISON:
1. Check your Grafana dashboard during test execution time
2. Compare the above metrics with Grafana panels
3. Throughput should match req/s shown in Grafana
4. Latency percentiles should be similar
5. Cache hit percentage should correlate
6. Availability should match uptime metrics

TIME RANGE FOR GRAFANA: ${new Date(data.state.testStartTimestamp).toISOString()} to ${new Date(data.state.testStartTimestamp + data.state.testRunDurationMs).toISOString()}
`,
    'summary.json': JSON.stringify(data, null, 2)
  };
  
  return summary;
}