import http from 'k6/http';
import { check, sleep } from 'k6';

// List your media asset URLs here (images/videos)
const mediaUrls = [
  'https://cdn.sohryuu.me/static/images/image_0.jpg',
  'https://cdn.sohryuu.me/static/images/image_1.jpg',
  'https://cdn.sohryuu.me/static/images/image_2.jpg',
  'https://cdn.sohryuu.me/static/images/image_3.jpg',
  'https://cdn.sohryuu.me/static/images/image_4.jpg',
  'https://cdn.sohryuu.me/static/images/image_5.jpg',
  'https://cdn.sohryuu.me/static/images/image_6.jpg',
  'https://cdn.sohryuu.me/static/images/image_7.jpg',
  'https://cdn.sohryuu.me/static/images/image_8.jpg',
  'https://cdn.sohryuu.me/static/images/image_9.jpg',
  // 'https://cdn.sohryuu.me/static/videos/video_0.mp4',
  // 'https://cdn.sohryuu.me/static/videos/video_1.mp4',
  // ...add more as needed
];

export let options = {
  vus: 10, // Number of virtual users
  duration: '30s', // Test duration
};

export default function () {
  // 1. Visit the main page
  let mainRes = http.get('https://tugas-akhir-458309.uc.r.appspot.com/images/trending/');
  check(mainRes, { 'main page loaded': (r) => r.status === 200 });

  // 2. Fetch each media asset (simulate browser loading images/videos)
  for (const url of mediaUrls) {
    let res = http.get(url);
    check(res, { 'media loaded': (r) => r.status === 200 });
    sleep(0.5); // Optional: simulate user "think time"
  }
}