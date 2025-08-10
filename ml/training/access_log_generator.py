import random
from datetime import datetime, timedelta
from collections import defaultdict
import csv

NUM_FILES = 101 
NUM_USERS = 50
TOTAL_DAYS = 30
REQUESTS_PER_DAY = 10000
START_TIME = datetime.now() - timedelta(days=TOTAL_DAYS)

CDN_DOMAIN = "https://cdn.sohryuu.me"
IMAGE_REFERRERS = [
    "/images/trending", "/images/popular", "/images/general", "/images/rare"
]
VIDEO_REFERRERS = [
    "/videos/short-clips", "/videos/documentaries", "/videos/tutorials", "/videos/archived"
]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
]

def generate_file_pool(num_files):
    half = num_files // 2
    return [
        f"/static/images/image_{i}.jpg" for i in range(half + num_files % 2)
    ] + [
        f"/static/videos/video_{i}.mp4" for i in range(half)
    ]

def assign_base_popularity(file_pool):
    return {f: random.randint(1, 10) for f in file_pool}

def schedule_trending_files(file_pool):
    return {
        2: random.sample(file_pool, 10),
        4: random.sample(file_pool, 10),
        6: random.sample(file_pool, 10),
    }

def create_log_entry(ip, timestamp, file_path, status, size, referrer, user_agent):
    return (
        f'{ip} - - [{timestamp.strftime("%d/%b/%Y:%H:%M:%S +0000")}] '
        f'"GET {file_path} HTTP/1.1" {status} {size} '
        f'"{referrer}" "{user_agent}"'
    )

file_pool = generate_file_pool(NUM_FILES)
base_popularity = assign_base_popularity(file_pool)
trending_schedule = schedule_trending_files(file_pool)

logs = []
daily_access_counts = defaultdict(lambda: [0] * TOTAL_DAYS)

for day in range(TOTAL_DAYS):
    current_day_time = START_TIME + timedelta(days=day)
    trending_today = trending_schedule.get(day, [])

    for _ in range(REQUESTS_PER_DAY):
        timestamp = current_day_time + timedelta(seconds=random.randint(0, 86399))
        ip = f"182.253.124.{random.randint(1, 254)}"

        # Weighted file selection
        weights = [
            base_popularity[f] * (5 if f in trending_today else 1)
            for f in file_pool
        ]
        file_path = random.choices(file_pool, weights=weights, k=1)[0]

        # Increment access count
        daily_access_counts[file_path][day] += 1

        # Determine status and referrer
        if file_path.endswith(".mp4"):
            status = 206
            referrer = f"{CDN_DOMAIN}{random.choice(VIDEO_REFERRERS)}"
        else:
            status = 200
            referrer = f"{CDN_DOMAIN}{random.choice(IMAGE_REFERRERS)}"

        user_agent = random.choice(USER_AGENTS)
        size = random.randint(100000, 5000000)

        # Create and store log entry
        log_line = create_log_entry(ip, timestamp, file_path, status, size, referrer, user_agent)
        logs.append(log_line)

with open("access.log", "w") as f:
    for line in logs:
        f.write(line + "\n")

print("access.log generated successfully.")