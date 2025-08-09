import http from 'k6/http';
import { check, sleep } from 'k6';

// --- Configuration ---
// IMPORTANT: Replace with your actual Cloud Run URL and CDN URL
const APP_BASE_URL = 'https://web-server-577176926733.us-central1.run.app';
const CDN_BASE_URL = 'https://cdn.sohryuu.me';

// List of image and video groups to simulate browsing
const IMAGE_GROUPS = ['trending', 'popular', 'general', 'rare'];
const VIDEO_GROUPS = ['short-clips', 'documentaries', 'tutorials', 'archived'];

// --- Test Options with Geographic Distribution ---
export const options = {
  // K6 Cloud geographic distribution using correct syntax
  cloud: {
    distribution: {
      // US regions - 40% of traffic
      'amazon:us:ashburn': { loadZone: 'amazon:us:ashburn', percent: 25 },
      'amazon:us:portland': { loadZone: 'amazon:us:portland', percent: 15 },
      
      // EU regions - 35% of traffic  
      'amazon:ie:dublin': { loadZone: 'amazon:ie:dublin', percent: 20 },
      'amazon:de:frankfurt': { loadZone: 'amazon:de:frankfurt', percent: 15 },
      
      // Asia regions - 25% of traffic
      'amazon:sg:singapore': { loadZone: 'amazon:sg:singapore', percent: 15 },
      'amazon:jp:tokyo': { loadZone: 'amazon:jp:tokyo', percent: 10 }
    }
  },
  
  scenarios: {
    // Scenario 1: Simulate users browsing main pages and image categories via CDN
    browse_main_and_images: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 }, // Ramp up to 50 VUs over 2 minutes
        { duration: '5m', target: 50 }, // Stay at 50 VUs for 5 minutes
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
    'http_req_duration': ['p(95)<2000'], // 95% of requests should be below 2000ms (adjusted for global latency)
    'http_req_failed': ['rate<0.01'],    // Less than 1% of requests should fail
    
    // Scenario-specific thresholds
    'http_req_duration{scenario:browse_images}': ['p(95)<1500'],
    'http_req_duration{scenario:browse_videos}': ['p(95)<2000'],
    'http_req_duration{scenario:uploads}': ['p(95)<3000'],
    'http_req_duration{scenario:health_check}': ['p(95)<1000'],
  },
};

// --- Functions for each Scenario ---

// Simulates browsing the home page and then random image categories
export function browseMainAndImages() {
  // 1. Visit the main home page (served by Cloud Run)
  let res = http.get(`${APP_BASE_URL}/`, {
    timeout: '30s', // Add timeout for better error handling
    tags: { request_type: 'homepage' },
  });
  
  if (!check(res, { 
    'home page loaded (app)': (r) => r.status === 200,
    'home page response time OK': (r) => r.timings.duration < 3000,
  })) {
    console.error(`Homepage request failed: ${res.status} - ${res.body}`);
  }
  
  sleep(Math.random() * 2 + 1); // Random sleep between 1-3 seconds

  // 2. Browse a random image category (served by CDN)
  const randomImageGroup = IMAGE_GROUPS[Math.floor(Math.random() * IMAGE_GROUPS.length)];
  res = http.get(`${CDN_BASE_URL}/images/${randomImageGroup}`, {
    timeout: '30s',
    tags: { request_type: 'image_category', category: randomImageGroup },
  });
  
  check(res, { 
    [`image category ${randomImageGroup} loaded (cdn)`]: (r) => r.status === 200,
    'image category response time OK': (r) => r.timings.duration < 2000,
  });
  
  sleep(Math.random() * 2 + 1); // Random sleep between 1-3 seconds

  // 3. Fetch a specific static image from CDN (simulate browser loading assets)
  const randomImageIndex = Math.floor(Math.random() * 10);
  res = http.get(`${CDN_BASE_URL}/static/images/image_${randomImageIndex}.jpg`, {
    timeout: '30s',
    tags: { request_type: 'static_image' },
  });
  
  check(res, { 
    'static image loaded (cdn)': (r) => r.status === 200,
    'static image response time OK': (r) => r.timings.duration < 1500,
  });
  
  sleep(Math.random() * 1 + 0.5); // Random sleep between 0.5-1.5 seconds
}

// Simulates browsing random video categories
export function browseVideos() {
  // Browse a random video category (served by CDN)
  const randomVideoGroup = VIDEO_GROUPS[Math.floor(Math.random() * VIDEO_GROUPS.length)];
  const res = http.get(`${CDN_BASE_URL}/videos/${randomVideoGroup}`, {
    timeout: '30s',
    tags: { request_type: 'video_category', category: randomVideoGroup },
  });
  
  check(res, { 
    [`video category ${randomVideoGroup} loaded (cdn)`]: (r) => r.status === 200,
    'video category response time OK': (r) => r.timings.duration < 2500,
  });
  
  sleep(Math.random() * 3 + 2); // Random sleep between 2-5 seconds (simulate watching)
}

// Simulates an upload process (simplified: just hitting the upload page)
export function simulateUploads() {
  // Go to the upload page (served by Cloud Run)
  let res = http.get(`${APP_BASE_URL}/upload`, {
    timeout: '30s',
    tags: { request_type: 'upload_page' },
  });
  
  if (!check(res, { 
    'upload page loaded (app)': (r) => r.status === 200,
    'upload page response time OK': (r) => r.timings.duration < 3000,
  })) {
    console.error(`Upload page request failed: ${res.status}`);
  }
  
  sleep(Math.random() * 5 + 3); // Random sleep between 3-8 seconds (simulate form filling)

  // NOTE: Actual file upload implementation would go here
  // For now, we're just simulating the upload page visit
  // If you want to simulate actual uploads, you would need to:
  // 1. Prepare test files in your K6 Cloud project
  // 2. Use http.file() to create file objects
  // 3. Send POST requests with multipart/form-data
}

// Simulates a health check request
export function runHealthCheck() {
  const res = http.get(`${APP_BASE_URL}/health`, {
    timeout: '15s',
    tags: { request_type: 'health_check' },
  });
  
  check(res, { 
    'health check ok': (r) => r.status === 200,
    'health check response time OK': (r) => r.timings.duration < 1000,
  });
  
  sleep(10); // Health checks happen less frequently
}

// Setup function (runs once per VU at the start)
export function setup() {
  console.log('Starting load test with geographic distribution');
  console.log('Regions: US (40%), EU (35%), Asia (25%)');
  console.log(`Target app: ${APP_BASE_URL}`);
  console.log(`CDN: ${CDN_BASE_URL}`);
}

// Teardown function (runs once at the end of the test)
export function teardown(data) {
  console.log('Load test completed');
}