import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from collections import OrderedDict, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import os
from urllib.parse import urlparse
import re

# Configuration
CONFIG = {
    "input_file": "preprocessed_access_log.csv",  # preprocessed CSV
    "model_prefix": "web_lrb_model",
    "look_ahead_window": 100,
    "cache_size": 100,
    "min_file_accesses": 2 
}

# Data Preprocessing Access Log
def preprocess_web_access_log(csv_path):
    print(f"📖 Reading web access log: {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"📊 Raw data: {len(df)} records")

    # Clean and prepare the data
    print("🧹 Cleaning data...")

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove entries without file_id or with invalid status codes
    df = df.dropna(subset=["file_id"])
    df = df[df["status"].isin([200, 304])]  # Only successful requests

    # Clean file_id and ensure it's a string
    df["file_id"] = df["file_id"].astype(str)
    df = df[df["file_id"] != "nan"]
    df = df[df["file_id"] != ""]

    # Extract additional features from the log
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
    df["log_size"] = np.log1p(df["size"])

    # Extract file extension if not already present
    if "file_type" not in df.columns or df["file_type"].isna().all():
        df["file_type"] = df["file_id"].apply(extract_file_extension)

    # Extract client information
    df["client_type"] = df["user_agent"].apply(extract_client_type)
    df["is_bot"] = df["user_agent"].apply(is_bot_request)

    # Remove bot traffic for better patterns
    df = df[df["is_bot"] == False]

    # Filter files with minimum access count
    file_counts = df["file_id"].value_counts()
    valid_files = file_counts[file_counts >= CONFIG["min_file_accesses"]].index
    df = df[df["file_id"].isin(valid_files)]

    # Sort by timestamp
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    print(f"✅ Clean data: {len(df)} records, {df['file_id'].nunique()} unique files")
    return df

def extract_file_extension(file_id):
    if pd.isna(file_id):
        return "unknown"

    # Handle URLs
    if file_id.startswith("http"):
        parsed = urlparse(file_id)
        file_id = parsed.path

    # Extract extension
    if "." in file_id:
        ext = file_id.split(".")[-1].lower()
        # Common web file extensions
        if ext in ["html", "htm", "php", "asp", "jsp"]:
            return "html"
        elif ext in ["jpg", "jpeg", "png", "gif", "webp", "svg"]:
            return "image"
        elif ext in ["css"]:
            return "css"
        elif ext in ["js"]:
            return "javascript"
        elif ext in ["pdf"]:
            return "pdf"
        elif ext in ["txt", "log"]:
            return "text"
        elif ext in ["xml", "json"]:
            return "data"
        else:
            return ext
    return "unknown"

def extract_client_type(user_agent):
    """Extract client type from user agent"""
    if pd.isna(user_agent):
        return "unknown"

    ua = user_agent.lower()
    if "mobile" in ua or "android" in ua or "iphone" in ua:
        return "mobile"
    elif "tablet" in ua or "ipad" in ua:
        return "tablet"
    elif "chrome" in ua or "firefox" in ua or "safari" in ua or "edge" in ua:
        return "desktop"
    else:
        return "other"

def is_bot_request(user_agent):
    if pd.isna(user_agent):
        return False

    bot_patterns = [
        "bot", "crawler", "spider", "scraper", "curl", "wget",
        "googlebot", "bingbot", "slurp", "duckduckbot", "baiduspider"
    ]

    ua = user_agent.lower()
    return any(pattern in ua for pattern in bot_patterns)

# Future-Aware Label Generation for Web Access Log
def generate_future_aware_labels(df, look_ahead_window=100, cache_size=100):
    print("🔮 Generating future-aware labels...")

    # Simulate web cache (LRU-based)
    cache = OrderedDict()

    # Track various labels for each access
    hit_labels = []
    next_access_distances = []
    future_reuse_probs = []
    will_be_accessed_in_n = []
    cache_states = []

    # Pre-compute future access positions for each file
    print("🔍 Computing future access patterns...")
    file_future_accesses = defaultdict(list)
    for idx, file_id in enumerate(df["file_id"]):
        file_future_accesses[file_id].append(idx)

    # Process each access
    for current_idx, row in df.iterrows():
        file_id = row["file_id"]

        # Current cache hit/miss
        if file_id in cache:
            hit_labels.append(1)
            cache.move_to_end(file_id)
        else:
            hit_labels.append(0)
            if len(cache) >= cache_size:
                evicted = cache.popitem(last=False)
            cache[file_id] = True

        # Store current cache state
        cache_states.append(len(cache))

        # Calculate future-aware labels
        future_accesses = [pos for pos in file_future_accesses[file_id] if pos > current_idx]

        if future_accesses:
            # Next access distance (steps until next access)
            next_distance = future_accesses[0] - current_idx
            next_access_distances.append(next_distance)

            # Future reuse probability (considering multiple future accesses)
            future_access_weights = []
            for future_idx in future_accesses[:5]:  # Consider next 5 accesses
                distance = future_idx - current_idx
                if distance <= look_ahead_window:
                    weight = 1.0 / distance  # Closer accesses have higher weight
                    future_access_weights.append(weight)

            future_reuse_prob = min(sum(future_access_weights), 1.0)
            future_reuse_probs.append(future_reuse_prob)

            # Will be accessed in next N steps
            will_be_accessed = 1 if next_distance <= look_ahead_window else 0
            will_be_accessed_in_n.append(will_be_accessed)
        else:
            # No future access within the dataset
            next_access_distances.append(look_ahead_window * 3)  # Large distance
            future_reuse_probs.append(0.0)  # No reuse
            will_be_accessed_in_n.append(0)  # Not accessed

        # Progress indicator
        if current_idx % 10000 == 0:
            print(f"   Processed {current_idx}/{len(df)} records...")

    # Add labels to dataframe
    df["hit"] = hit_labels
    df["next_access_distance"] = next_access_distances
    df["future_reuse_prob"] = future_reuse_probs
    df["will_be_accessed_in_n"] = will_be_accessed_in_n
    df["cache_state"] = cache_states

    print(f"✅ Labels generated successfully!")
    print(f"   Hit rate: {np.mean(hit_labels):.3f}")
    print(f"   Avg next access distance: {np.mean(next_access_distances):.1f}")
    print(f"   Avg future reuse prob: {np.mean(future_reuse_probs):.3f}")

    return df

# Enhanced Feature Generation for Access Log
def generate_web_features(df):
    print("🔧 Generating web-specific features...")

    # Basic access patterns
    access_counts = {}
    access_count_vals = []
    for file_id in df["file_id"]:
        access_count_vals.append(access_counts.get(file_id, 0))
        access_counts[file_id] = access_counts.get(file_id, 0) + 1
    df["access_count"] = access_count_vals

    # Time-based features
    last_seen = {}
    ages = []
    for idx, row in df.iterrows():
        file_id = row["file_id"]
        timestamp = row["timestamp"]
        last_time = last_seen.get(file_id)
        age = (timestamp - last_time).total_seconds() if last_time else -1
        ages.append(age)
        last_seen[file_id] = timestamp
    df["age_since_last_access"] = ages

    # Temporal patterns
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hours"] = df["hour"].between(9, 17).astype(int)

    # Categorical encodings
    df["file_type_encoded"] = df["file_type"].astype("category").cat.codes
    df["client_type_encoded"] = df["client_type"].astype("category").cat.codes

    # Advanced features
    # Recency score (inverse of age)
    df["recency_score"] = 1.0 / (1.0 + df["age_since_last_access"].fillna(0) / 3600)  # Hours

    # Frequency score (log of access count)
    df["frequency_score"] = np.log1p(df["access_count"])

    # Popularity score (global frequency of the file)
    file_popularity = df["file_id"].value_counts().to_dict()
    df["popularity_score"] = df["file_id"].map(file_popularity)
    df["log_popularity"] = np.log1p(df["popularity_score"])

    # Size-based features
    df["size_category"] = pd.cut(df["size"], bins=[0, 1024, 10240, 102400, float('inf')],
                                labels=["small", "medium", "large", "very_large"])
    df["size_category_encoded"] = df["size_category"].astype("category").cat.codes

    # Combined features
    df["access_pattern_score"] = df["recency_score"] * df["frequency_score"]
    df["temporal_score"] = df["is_business_hours"] * 0.5 + (1 - df["is_weekend"]) * 0.3
    df["content_score"] = df["log_size"] * 0.3 + df["log_popularity"] * 0.7

    # Filter out entries with no valid age
    df = df[df["age_since_last_access"] >= 0]

    print(f"✅ Features generated: {len(df)} records after filtering")
    return df

# Model Training
def train_web_lrb_models(df, output_prefix="web_lrb_model"):
    print("🎯 Training web-specific LRB models...")

    # Define feature columns
    feature_cols = [
        "access_count", "age_since_last_access", "file_type_encoded",
        "client_type_encoded", "log_size", "size_category_encoded",
        "recency_score", "frequency_score", "popularity_score",
        "access_pattern_score", "temporal_score", "content_score",
        "hour", "day_of_week", "is_weekend", "is_business_hours",
        "cache_state"
    ]

    # Prepare training data
    X = df[feature_cols]

    # Split data
    X_train, X_val, df_train, df_val = train_test_split(X, df, test_size=0.2, random_state=42)

    models = {}

    # 1. Binary classification: Will be accessed in next N steps
    print("\n🎯 Training binary classifier...")
    y_binary = df_train["will_be_accessed_in_n"]
    model_binary = lgb.LGBMClassifier(
        random_state=42,
        verbose=-1,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    model_binary.fit(X_train, y_binary)

    y_pred_binary = model_binary.predict(X_val)
    y_prob_binary = model_binary.predict_proba(X_val)[:, 1]

    print("📊 Binary Classification Report:")
    print(classification_report(df_val["will_be_accessed_in_n"], y_pred_binary))

    # Feature importance
    feature_importance = model_binary.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\n🔝 Top 10 Important Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

    models["binary"] = model_binary

    # 2. Regression: Next access distance
    print("\n🎯 Training distance regression model...")
    y_distance = df_train["next_access_distance"]
    model_distance = lgb.LGBMRegressor(
        random_state=42,
        verbose=-1,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    model_distance.fit(X_train, y_distance)

    y_pred_distance = model_distance.predict(X_val)
    mse = mean_squared_error(df_val["next_access_distance"], y_pred_distance)
    mae = mean_absolute_error(df_val["next_access_distance"], y_pred_distance)
    print(f"📊 Distance Prediction - MSE: {mse:.2f}, MAE: {mae:.2f}")

    models["distance"] = model_distance

    # 3. Regression: Future reuse probability
    print("\n🎯 Training reuse probability model...")
    y_reuse = df_train["future_reuse_prob"]
    model_reuse = lgb.LGBMRegressor(
        random_state=42,
        verbose=-1,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    model_reuse.fit(X_train, y_reuse)

    y_pred_reuse = model_reuse.predict(X_val)
    mse_reuse = mean_squared_error(df_val["future_reuse_prob"], y_pred_reuse)
    mae_reuse = mean_absolute_error(df_val["future_reuse_prob"], y_pred_reuse)
    print(f"📊 Reuse Probability - MSE: {mse_reuse:.4f}, MAE: {mae_reuse:.4f}")

    models["reuse"] = model_reuse

    # Save all models
    print("\n💾 Saving models...")
    for name, model in models.items():
        filename = f"{output_prefix}_{name}.pkl"
        joblib.dump(model, filename)
        print(f"✅ {name.title()} model saved to {filename}")

    # Save feature columns for later use
    feature_info = {
        "feature_columns": feature_cols,
        "model_info": {
            "binary_accuracy": np.mean(y_pred_binary == df_val["will_be_accessed_in_n"]),
            "distance_mae": mae,
            "reuse_mae": mae_reuse
        }
    }

    with open(f"{output_prefix}_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    print(f"✅ Model info saved to {output_prefix}_info.json")

    return models, feature_cols

# LRB Eviction Scorer
class WebLRBEvictionScorer:
    def __init__(self, model_prefix="web_lrb_model"):
        self.models = {}
        self.feature_columns = []

        try:
            # Load models
            self.models["binary"] = joblib.load(f"{model_prefix}_binary.pkl")
            self.models["distance"] = joblib.load(f"{model_prefix}_distance.pkl")
            self.models["reuse"] = joblib.load(f"{model_prefix}_reuse.pkl")

            # Load feature info
            with open(f"{model_prefix}_info.json", "r") as f:
                info = json.load(f)
                self.feature_columns = info["feature_columns"]

            print(f"✅ Loaded web LRB models: {model_prefix}")
            print(f"📊 Features: {len(self.feature_columns)}")

        except FileNotFoundError as e:
            print(f"⚠️ Model file not found: {e}")

    def score_for_eviction(self, features):
        if not self.models:
            return np.random.random(len(features))

        scores = np.zeros(len(features))

        # Ensure features are in the correct order
        if isinstance(features, pd.DataFrame):
            features = features[self.feature_columns]

        # Binary prediction - invert for eviction scoring
        if "binary" in self.models:
            access_prob = self.models["binary"].predict_proba(features)[:, 1]
            scores += (1.0 - access_prob) * 0.4  # 40% weight

        # Distance prediction - normalize and use for eviction
        if "distance" in self.models:
            distances = self.models["distance"].predict(features)
            # Normalize distances to [0, 1] range
            max_distance = 300  # Adjust based on your look-ahead window
            normalized_distances = np.clip(distances / max_distance, 0, 1)
            scores += normalized_distances * 0.3  # 30% weight

        # Reuse probability - invert for eviction
        if "reuse" in self.models:
            reuse_probs = self.models["reuse"].predict(features)
            scores += (1.0 - np.clip(reuse_probs, 0, 1)) * 0.3  # 30% weight

        return scores

    def recommend_eviction(self, cache_items, features, n_evict=1):
        """
        Recommend web cache items to evict
        """
        scores = self.score_for_eviction(features)
        evict_indices = np.argsort(scores)[-n_evict:][::-1]

        recommendations = []
        for i in evict_indices:
            recommendations.append({
                "item": cache_items[i],
                "eviction_score": scores[i],
                "index": i
            })

        return recommendations

# Main Training Pipeline
def train_web_lrb_pipeline(csv_path, model_prefix="web_lrb_model"):
    print("🌟 Starting Web LRB Training Pipeline")
    print("=" * 60)

    # Step 1: Preprocess data
    df = preprocess_web_access_log(csv_path)

    # Step 2: Generate future-aware labels
    df = generate_future_aware_labels(df, CONFIG["look_ahead_window"], CONFIG["cache_size"])

    # Step 3: Generate features
    df = generate_web_features(df)

    # Step 4: Train models
    models, feature_cols = train_web_lrb_models(df, model_prefix)

    # Step 5: Create scorer
    scorer = WebLRBEvictionScorer(model_prefix)

    # Step 6: Test with sample data
    print("\n🧪 Testing with sample data...")
    if len(df) > 10:
        sample_df = df.head(10)
        sample_items = sample_df["file_id"].tolist()
        sample_features = sample_df[feature_cols]

        recommendations = scorer.recommend_eviction(sample_items, sample_features, n_evict=3)

        print("✅ Sample eviction recommendations:")
        for rec in recommendations:
            print(f"   File: {rec['item'][:50]}... Score: {rec['eviction_score']:.3f}")

    print(f"\n🎉 Training complete!")
    print(f"📁 Models saved with prefix: {model_prefix}")

    return models, scorer, df

# Easy execution
def train_with_access_log(csv_path="access.log"):
    return train_web_lrb_pipeline(csv_path, CONFIG["model_prefix"])

# Main execution
if __name__ == "__main__":
    # Train with your access log
    models, scorer, df = train_with_access_log(CONFIG["input_file"])
    print("\n💡 Usage:")
    print("# Train with your access log:")
    print("models, scorer, df = train_with_access_log('access_log.csv')")
    print("\n# Use for cache eviction:")
    print("recommendations = scorer.recommend_eviction(cache_items, features, n_evict=5)")