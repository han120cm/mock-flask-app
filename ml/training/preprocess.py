import re
import csv
from datetime import datetime

ACCESS_LOG_PATH = "access.log"  
OUTPUT_CSV_PATH = "preprocessed_access_log.csv"  

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".mp4", ".webm", ".mkv"]
TYPE_BY_EXT = {
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
    ".mp4": "video", ".webm": "video", ".mkv": "video"
}

def parse_nginx_log(line):
    match = re.match(r'([\d.]+) - - \[([^\]]+)\] "(\w+) ([^ ]+) HTTP/[\d.]+" (\d{3}) (\d+) "([^"]*)" "([^"]*)"', line)
    if not match:
        return None
    ip, time_str, method, path, status, size, referrer, user_agent = match.groups()

    ext = path.split('.')[-1].lower()
    file_type = TYPE_BY_EXT.get(f".{ext}", "other")

    if file_type == "other":
        return None  # Skip non-media files

    try:
        timestamp = datetime.strptime(time_str, "%d/%b/%Y:%H:%M:%S %z").isoformat()
    except ValueError:
        return None

    file_id = path.split("/")[-1].split('.')[0]  

    return [ip, timestamp, path, file_id, file_type, int(status), int(size), referrer, user_agent]


rows = []
with open(ACCESS_LOG_PATH, "r") as f:
    for line in f:
        row = parse_nginx_log(line)
        if row:
            rows.append(row)

with open(OUTPUT_CSV_PATH, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["ip", "timestamp", "url", "file_id", "file_type", "status", "size", "referrer", "user_agent"])
    writer.writerows(rows)

print(f"Saved {len(rows)} entries to {OUTPUT_CSV_PATH}")