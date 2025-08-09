import http from 'k6/http';
import { check, sleep } from 'k6';

// --- Configuration ---
// IMPORTANT: Replace with your actual Cloud Run URL
const APP_BASE_URL = 'https://web-server-577176926733.us-central1.run.app';
const BUCKET_BASE_URL = 'https://storage.googleapis.com/bucket-main-ta';

// List of image and video groups to simulate browsing
const IMAGE_GROUPS = ['trending', 'popular', 'general', 'rare'];
const VIDEO_GROUPS = ['short-clips', 'documentaries', 'tutorials', 'archived'];

// --- Test Options ---
export const options = {
  scenarios: {
    // Scenario 1: Simulate users browsing main pages and image categories from the app server
    browse_main_and_images: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 }, // Ramp up to 50 VUs over 2 minutes
        { duration: '5m', target: 50 }, // Stay at 50 VUs for 5 minutes
        { duration: '2m', target: 0 },  // Ramp down to 0 VUs over 2 minutes
      ],
      exec: 'browseMainAndImages',
      tags: { scenario: 'browse_images_no_cdn' },
    },
    
    // Scenario 2: Simulate users browsing video categories from the app server
    browse_videos: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 20 }, // Ramp up to 20 VUs over 2 minutes
        { duration: '5m', target: 20 }, // Stay at 20 VUs for 5 minutes
        { duration: '2m', target: 0 },  // Ramp down to 0 VUs over 2 minutes
      ],
      exec: 'browseVideos',
      tags: { scenario: 'browse_videos_no_cdn' },
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
    'http_req_duration': ['p(95)<3000'], // 95% of requests should be below 3000ms (higher threshold without CDN)
    'http_req_failed': ['rate<0.01'],    // Less than 1% of requests should fail
    
    // Scenario-specific thresholds
    'http_req_duration{scenario:browse_images_no_cdn}': ['p(95)<2500'],
    'http_req_duration{scenario:browse_videos_no_cdn}': ['p(95)<3000'],
    'http_req_duration{scenario:uploads}': ['p(95)<3000'],
    'http_req_duration{scenario:health_check}': ['p(95)<1000'],
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
      scenario: 'browse_images_no_cdn',
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
  
  sleep(Math.random() * 2 + 1);

  // 2. Browse a random image category (served by App Server)
  const randomImageGroup = IMAGE_GROUPS[Math.floor(Math.random() * IMAGE_GROUPS.length)];
  res = http.get(`${APP_BASE_URL}/images/${randomImageGroup}`, {
    timeout: '30s',
    tags: { 
      request_type: 'image_category', 
      category: randomImageGroup,
      scenario: 'browse_images_no_cdn',
      source: 'app_server'
    },
  });
  
  check(res, { 
    [`image category ${randomImageGroup} loaded (app)`]: (r) => r.status === 200,
    'image category response time OK': (r) => r.timings.duration < 2500,
  });
  
  sleep(Math.random() * 2 + 1);

  // 3. Fetch a specific static image from GCS Bucket
  const randomImageIndex = Math.floor(Math.random() * 10); // trending group has 0-9
  res = http.get(`${BUCKET_BASE_URL}/static/images/image_${randomImageIndex}.jpg`, {
    timeout: '30s',
    tags: { 
      request_type: 'static_image',
      scenario: 'browse_images_no_cdn',
      source: 'gcs_bucket',
      image_index: randomImageIndex.toString()
    },
  });
  
  check(res, { 
    'static image request completed': (r) => r.status === 200,
    'static image response time measured': (r) => r.timings.duration > 0,
  });
  
  sleep(Math.random() * 1 + 0.5);
}

// Simulates browsing random video categories
export function browseVideos() {
  // 1. Browse a random video category (served by App Server)
  const randomVideoGroup = VIDEO_GROUPS[Math.floor(Math.random() * VIDEO_GROUPS.length)];
  let res = http.get(`${APP_BASE_URL}/videos/${randomVideoGroup}`, {
    timeout: '30s',
    tags: { 
      request_type: 'video_category', 
      category: randomVideoGroup,
      scenario: 'browse_videos_no_cdn',
      source: 'app_server'
    },
  });
  
  check(res, { 
    [`video category ${randomVideoGroup} loaded (app)`]: (r) => r.status === 200,
    'video category response time OK': (r) => r.timings.duration < 3000,
  });
  
  sleep(Math.random() * 3 + 2);

  // 2. Fetch a specific static video from GCS Bucket
  const randomVideoIndex = Math.floor(Math.random() * 10); // short-clips group has 0-9
  res = http.get(`${BUCKET_BASE_URL}/static/videos/video_${randomVideoIndex}.mp4`, {
    timeout: '45s', // Longer timeout for video files
    tags: {
      request_type: 'static_video',
      scenario: 'browse_videos_no_cdn',
      source: 'gcs_bucket',
      video_index: randomVideoIndex.toString()
    },
  });

  check(res, {
    'static video request completed': (r) => r.status === 200,
    'static video response time measured': (r) => r.timings.duration > 0,
  });

  sleep(Math.random() * 2 + 1);
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
  });
  
  if (!uploadCheck) {
    console.error(`Upload page request failed: ${res.status}`);
  }
  
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
  });
  
  sleep(10);
}

// Setup function (runs once per VU at the start)
export function setup() {
  console.log('Starting load test for App Server (No CDN)');
  console.log(`Target app: ${APP_BASE_URL}`);
  console.log(`Content source: ${BUCKET_BASE_URL}`);
  
  // Test connectivity to the app server
  console.log('Testing connectivity...');
  
  try {
    const appTest = http.get(`${APP_BASE_URL}/health`, { timeout: '10s' });
    console.log(`App server connectivity: ${appTest.status} (${appTest.timings.duration}ms)`);
  } catch (e) {
    console.error(`App server connectivity failed: ${e}`);
  }
}

// Teardown function (runs once at the end of the test)
export function teardown(data) {
  console.log('Load test completed');
}
