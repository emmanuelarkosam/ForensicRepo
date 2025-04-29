
# Dataset Documentation

Synthetic datasets replicating original study schema:

1. `synthetic_posts.csv`:
   - Columns: `id` (int), `platform` (Facebook/Twitter/Instagram), `text` (string), `latitude` (float), `longitude` (float), `timestamp` (ISO-8601)

2. `synthetic_image_features.csv`:
   - Columns: `id` (int), `feat1`...`feat10` (float) representing pre-extracted image features, `label` (0/1 binary classification)

3. `synthetic_tweets.json`:
   - List of objects with fields: `id`, `user`, `text`, `retweet_count`, `hashtags` (list), `timestamp`.
