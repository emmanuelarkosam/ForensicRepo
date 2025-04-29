
# Preprocessing Steps

## Text Data
1. Lowercase all text.
2. Tokenization and Stopword removal via `TfidfVectorizer(stop_words='english')`.
3. TF-IDF vectorization.
4. Train-test split (80/20) with fixed `random_state=42`.

## Image Features
1. Standardize each feature to zero mean, unit variance.
2. Split (80/20) with same `random_state=42`.

## Tweets Metadata
- Parse JSON, extract `retweet_count` and number of `hashtags` as numeric features.
- Merge with text and image datasets if needed.
