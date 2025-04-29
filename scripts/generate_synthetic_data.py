
"""Generate synthetic datasets mirroring study schema."""
import random, csv, json, pathlib
from datetime import datetime, timedelta

# Parameters
N_POSTS = 1000
N_IMAGES = 500
N_TWEETS = 500

# 1. synthetic_posts.csv
out_posts = pathlib.Path("datasets/synthetic_posts.csv")
out_posts.parent.mkdir(exist_ok=True)
platforms = ["Facebook", "Twitter", "Instagram"]
with out_posts.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "platform", "text", "latitude", "longitude", "timestamp"])
    for i in range(N_POSTS):
        plat = random.choice(platforms)
        text = f"Sample post content {i}"
        lat = round(random.uniform(-90,90),6)
        lon = round(random.uniform(-180,180),6)
        ts = (datetime.utcnow() - timedelta(days=random.randint(0,365))).isoformat()
        writer.writerow([i, plat, text, lat, lon, ts])

# 2. synthetic_image_features.csv
out_img = pathlib.Path("datasets/synthetic_image_features.csv")
with out_img.open("w", newline="") as f:
    writer = csv.writer(f)
    header = ["id"] + [f"feat{j}" for j in range(1,11)] + ["label"]
    writer.writerow(header)
    for i in range(N_IMAGES):
        feats = [round(random.uniform(0,1),4) for _ in range(10)]
        label = random.randint(0,1)
        writer.writerow([i] + feats + [label])

# 3. synthetic_tweets.json
out_tweets = pathlib.Path("datasets/synthetic_tweets.json")
with out_tweets.open("w") as f:
    data = []
    for i in range(N_TWEETS):
        data.append({
            "id": i,
            "user": f"user_{random.randint(1,100)}",
            "text": f"Tweet text {i}",
            "retweet_count": random.randint(0,50),
            "hashtags": random.sample(["#forensic","#social","#ai","#ml"], k=random.randint(0,3)),
            "timestamp": (datetime.utcnow() - timedelta(days=random.randint(0,365))).isoformat()
        })
    json.dump(data, f, indent=2)

print("Synthetic datasets generated in datasets/") 
