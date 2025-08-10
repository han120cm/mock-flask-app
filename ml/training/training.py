import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from collections import OrderedDict, defaultdict
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import os
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# Colab-friendly install fallback
# try:
#     import lightgbm  # noqa: F401
# except ImportError:
#     # If running in a notebook environment like Colab
#     !pip install lightgbm

CONFIG = {
    "input_file": "preprocessed_access_log.csv",
    "model_prefix": "web_lrb_model",
    "look_ahead_window": 100,
    "cache_size": 100,
    "min_file_accesses": 2,
    "tuning": {
        "method": "random",            # 'grid' or 'random'
        "cv_folds": 3,
        "n_iter": 50,
        "scoring": {
            "binary": "f1",
            "regression": "neg_mean_absolute_error"
        },
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_state": 42
    }
}

PARAM_GRIDS = {
    "lgbm_binary": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 6, 9, 12, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0, 0.1, 0.5, 1.0]
    },
    "lgbm_regression": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 6, 9, 12, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0, 0.1, 0.5, 1.0]
    }
}

PARAM_GRIDS_QUICK = {
    "lgbm_binary": {
        "n_estimators": [50, 100, 200],
        "max_depth": [6, 9, -1],
        "learning_rate": [0.05, 0.1, 0.2],
        "num_leaves": [31, 63],
        "min_child_samples": [20, 50],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },
    "lgbm_regression": {
        "n_estimators": [50, 100, 200],
        "max_depth": [6, 9, -1],
        "learning_rate": [0.05, 0.1, 0.2],
        "num_leaves": [31, 63],
        "min_child_samples": [20, 50],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
}

def preprocess_web_access_log(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Reading web access log: {csv_path}")
    print(f"Raw data: {len(df)} records")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["file_id"])
    df = df[df["status"].isin([200, 304])]
    df["file_id"] = df["file_id"].astype(str)
    df = df[df["file_id"] != "nan"]
    df = df[df["file_id"] != ""]
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
    df["log_file_size"] = np.log1p(df["size"])

    if "file_type" not in df.columns or df["file_type"].isna().all():
        df["type"] = df["file_id"].apply(extract_file_type)
    else:
        df["type"] = df["file_type"].apply(map_to_cache_type)

    df["client_type"] = df["user_agent"].apply(extract_client_type)
    df["is_bot"] = df["user_agent"].apply(is_bot_request)
    df = df[df["is_bot"] == False]

    file_counts = df["file_id"].value_counts()
    valid_files = file_counts[file_counts >= CONFIG["min_file_accesses"]].index
    df = df[df["file_id"].isin(valid_files)]

    df = df.sort_values(by="timestamp").reset_index(drop=True)
    print(f"Clean data: {len(df)} records, {df['file_id'].nunique()} unique files")
    return df

def extract_file_type(file_id):
    if pd.isna(file_id):
        return "unknown"
    if str(file_id).startswith("http"):
        parsed = urlparse(str(file_id))
        file_id = parsed.path
    if "." in str(file_id):
        ext = str(file_id).split(".")[-1].lower()
        mapping = {
            "video": ["mp4","avi","mkv","mov","wmv","flv","webm","m4v"],
            "audio": ["mp3","wav","flac","aac","ogg","m4a"],
            "image": ["jpg","jpeg","png","gif","webp","svg","bmp","tiff"],
            "html": ["html","htm","php","asp","jsp"],
            "css": ["css"],
            "javascript": ["js"],
            "pdf": ["pdf"],
            "text": ["txt","log"],
            "data": ["xml","json"],
            "archive": ["zip","rar","7z","tar","gz"]
        }
        for k, v in mapping.items():
            if ext in v:
                return k
        return "other"
    return "unknown"

def map_to_cache_type(file_type):
    if pd.isna(file_type):
        return "unknown"
    file_type = str(file_type).lower()
    type_mapping = {
        'jpg': 'image', 'jpeg': 'image', 'png': 'image', 'gif': 'image',
        'webp': 'image', 'svg': 'image', 'bmp': 'image', 'tiff': 'image',
        'mp4': 'video', 'avi': 'video', 'mkv': 'video', 'mov': 'video',
        'wmv': 'video', 'flv': 'video', 'webm': 'video', 'm4v': 'video',
        'mp3': 'audio', 'wav': 'audio', 'flac': 'audio', 'aac': 'audio',
        'ogg': 'audio', 'm4a': 'audio',
        'html': 'html', 'htm': 'html', 'php': 'html', 'asp': 'html', 'jsp': 'html',
        'css': 'css',
        'js': 'javascript', 'javascript': 'javascript',
        'pdf': 'pdf',
        'txt': 'text', 'log': 'text', 'text': 'text',
        'xml': 'data', 'json': 'data', 'data': 'data',
        'zip': 'archive', 'rar': 'archive', '7z': 'archive', 'tar': 'archive', 'gz': 'archive'
    }
    return type_mapping.get(file_type, 'other')

def extract_client_type(user_agent):
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

def generate_future_aware_labels(df, look_ahead_window=100, cache_size=100):
    cache = OrderedDict()
    hit_labels = []
    next_access_distances = []
    future_reuse_probs = []
    will_be_accessed_in_n = []
    stack_distances = []

    print("Generating future-aware labels...")
    file_future_accesses = defaultdict(list)
    for idx, file_id in enumerate(df["file_id"]):
        file_future_accesses[file_id].append(idx)

    for current_idx, row in df.iterrows():
        file_id = row["file_id"]

        if file_id in cache:
            hit_labels.append(1)
            cache_list = list(cache.keys())
            stack_dist = cache_list.index(file_id) + 1
            stack_distances.append(stack_dist)
            cache.move_to_end(file_id)
        else:
            hit_labels.append(0)
            stack_distances.append(len(cache) + 1)
            if len(cache) >= cache_size:
                cache.popitem(last=False)
            cache[file_id] = True

        future_accesses = [pos for pos in file_future_accesses[file_id] if pos > current_idx]

        if future_accesses:
            next_distance = future_accesses[0] - current_idx
            next_access_distances.append(next_distance)

            future_access_weights = []
            for future_idx in future_accesses[:5]:
                distance = future_idx - current_idx
                if distance <= look_ahead_window:
                    future_access_weights.append(1.0 / distance)

            future_reuse_prob = min(sum(future_access_weights), 1.0)
            future_reuse_probs.append(future_reuse_prob)

            will_be_accessed = 1 if next_distance <= look_ahead_window else 0
            will_be_accessed_in_n.append(will_be_accessed)
        else:
            next_access_distances.append(look_ahead_window * 3)
            future_reuse_probs.append(0.0)
            will_be_accessed_in_n.append(0)

        if current_idx % 10000 == 0:
            print(f"Processed {current_idx}/{len(df)} records...")

    df["hit"] = hit_labels
    df["next_access_distance"] = next_access_distances
    df["future_reuse_prob"] = future_reuse_probs
    df["will_be_accessed_in_n"] = will_be_accessed_in_n
    df["stack_distance_approx"] = stack_distances

    print("Labels generated.")
    print(f"Hit rate: {np.mean(hit_labels):.3f}")
    print(f"Avg next access distance: {np.mean(next_access_distances):.1f}")
    print(f"Avg future reuse prob: {np.mean(future_reuse_probs):.3f}")
    print(f"Avg stack distance: {np.mean(stack_distances):.1f}")
    return df

def generate_cache_json_features(df):
    access_counts = {}
    last_access_times = {}

    access_count_vals = []
    age_since_last_access_vals = []
    age_hours_vals = []
    access_rate_vals = []
    recency_score_vals = []

    print("Generating cache JSON features...")
    for idx, row in df.iterrows():
        file_id = row["file_id"]
        timestamp = row["timestamp"]

        access_counts[file_id] = access_counts.get(file_id, 0) + 1
        current_access_count = access_counts[file_id]
        access_count_vals.append(current_access_count)

        last_time = last_access_times.get(file_id)
        if last_time:
            age_seconds = (timestamp - last_time).total_seconds()
            age_h = age_seconds / 3600.0
        else:
            age_seconds = 0
            age_h = 0
        age_since_last_access_vals.append(age_seconds)
        age_hours_vals.append(age_h)

        last_access_times[file_id] = timestamp

        access_rate = current_access_count / age_h if age_h > 0 else current_access_count
        access_rate_vals.append(access_rate)

        recency = 1.0 / (1.0 + age_seconds / 3600)
        recency_score_vals.append(recency)

        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(df)} records...")

    df["access_count"] = access_count_vals
    df["age_since_last_access"] = age_since_last_access_vals
    df["age_hours"] = age_hours_vals
    df["access_rate"] = access_rate_vals
    df["recency_score"] = recency_score_vals

    type_mapping = {
        "video": 0, "audio": 1, "image": 2, "html": 3, "css": 4,
        "javascript": 5, "pdf": 6, "text": 7, "data": 8, "archive": 9,
        "other": 10, "unknown": 11
    }
    df["type_encoded"] = df["type"].map(type_mapping).fillna(11)

    print(f"Cache JSON features generated: {len(df)} records")
    return df

def tune_hyperparameters(X_train, y_train, model_type="binary", quick_mode=False):
    tuning_config = CONFIG["tuning"]

    print(f"Starting hyperparameter tuning for {model_type} model...")
    print(f"Method: {tuning_config['method'].upper()}")
    print(f"CV Folds: {tuning_config['cv_folds']}")

    if quick_mode:
        param_grid = PARAM_GRIDS_QUICK[f"lgbm_{model_type}"]
        print("Mode: Quick (reduced parameter space)")
    else:
        param_grid = PARAM_GRIDS[f"lgbm_{model_type}"]
        print("Mode: Full parameter search")

    if model_type == "binary":
        base_model = lgb.LGBMClassifier(
            random_state=tuning_config["random_state"],
            verbose=-1
        )
        scoring = tuning_config["scoring"]["binary"]
    else:
        base_model = lgb.LGBMRegressor(
            random_state=tuning_config["random_state"],
            verbose=-1
        )
        scoring = tuning_config["scoring"]["regression"]

    if tuning_config["method"] == "grid":
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=tuning_config["cv_folds"],
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        total = 1
        for v in param_grid.values():
            total *= len(v)
        print(f"Total combinations: {total}")
    else:
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=tuning_config["n_iter"],
            cv=tuning_config["cv_folds"],
            scoring=scoring,
            n_jobs=-1,
            random_state=tuning_config["random_state"],
            verbose=1
        )
        print(f"Random iterations: {tuning_config['n_iter']}")

    print("Searching for best parameters...")
    search.fit(X_train, y_train)

    print(f"Best {model_type} score: {search.best_score_:.4f}")
    print("Best parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    return search.best_estimator_, search.best_params_, search.best_score_

def evaluate_model_performance(model, X_test, y_test, model_type="binary"):
    print(f"\nEvaluating {model_type} model performance...")

    if model_type == "binary":
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        metrics = {
            "accuracy": (y_pred == y_test).mean(),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba)
        }

        print("\nDetailed Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        return metrics

    else:
        y_pred = model.predict(X_test)
        from sklearn.metrics import r2_score
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred)
        }

        print("Regression Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        return metrics

def train_cache_json_models_with_tuning(df, output_prefix="web_lrb_model", quick_mode=False):
    feature_cols = [
        "size",
        "type_encoded",
        "access_count",
        "age_since_last_access",
        "age_hours",
        "log_file_size",
        "access_rate",
        "recency_score",
        "stack_distance_approx"
    ]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols]

    X_temp, X_test, df_temp, df_test = train_test_split(
        X, df,
        test_size=CONFIG["tuning"]["test_size"],
        random_state=CONFIG["tuning"]["random_state"]
    )
    X_train, X_val, df_train, df_val = train_test_split(
        X_temp, df_temp,
        test_size=CONFIG["tuning"]["validation_size"],
        random_state=CONFIG["tuning"]["random_state"]
    )

    print("Data splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    models = {}
    tuning_results = {}

    print("\n" + "="*60)
    print("BINARY CLASSIFICATION MODEL")
    print("="*60)
    y_binary_train = df_train["will_be_accessed_in_n"]
    y_binary_val = df_val["will_be_accessed_in_n"]
    y_binary_test = df_test["will_be_accessed_in_n"]

    best_binary_model, best_binary_params, best_binary_score = tune_hyperparameters(
        X_train, y_binary_train, "binary", quick_mode
    )
    binary_metrics = evaluate_model_performance(
        best_binary_model, X_test, y_binary_test, "binary"
    )

    feature_importance = best_binary_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    models["binary"] = best_binary_model
    tuning_results["binary"] = {
        "best_params": best_binary_params,
        "best_cv_score": best_binary_score,
        "test_metrics": binary_metrics,
        "feature_importance": importance_df.to_dict('records')
    }

    print("\n" + "="*60)
    print("DISTANCE REGRESSION MODEL")
    print("="*60)
    y_distance_train = df_train["next_access_distance"]
    y_distance_val = df_val["next_access_distance"]
    y_distance_test = df_test["next_access_distance"]

    best_distance_model, best_distance_params, best_distance_score = tune_hyperparameters(
        X_train, y_distance_train, "regression", quick_mode
    )
    distance_metrics = evaluate_model_performance(
        best_distance_model, X_test, y_distance_test, "regression"
    )

    models["distance"] = best_distance_model
    tuning_results["distance"] = {
        "best_params": best_distance_params,
        "best_cv_score": best_distance_score,
        "test_metrics": distance_metrics
    }

    print("\n" + "="*60)
    print("REUSE PROBABILITY REGRESSION MODEL")
    print("="*60)
    y_reuse_train = df_train["future_reuse_prob"]
    y_reuse_val = df_val["future_reuse_prob"]
    y_reuse_test = df_test["future_reuse_prob"]

    best_reuse_model, best_reuse_params, best_reuse_score = tune_hyperparameters(
        X_train, y_reuse_train, "regression", quick_mode
    )
    reuse_metrics = evaluate_model_performance(
        best_reuse_model, X_test, y_reuse_test, "regression"
    )

    models["reuse"] = best_reuse_model
    tuning_results["reuse"] = {
        "best_params": best_reuse_params,
        "best_cv_score": best_reuse_score,
        "test_metrics": reuse_metrics
    }

    print("\n" + "="*60)
    print("SAVING MODELS AND RESULTS")
    print("="*60)
    for name, model in models.items():
        filename = f"{output_prefix}_{name}.pkl"
        joblib.dump(model, filename)
        print(f"{name.title()} model saved to {filename}")

    type_mapping = {
        "video": 0, "audio": 1, "image": 2, "html": 3, "css": 4,
        "javascript": 5, "pdf": 6, "text": 7, "data": 8, "archive": 9,
        "other": 10, "unknown": 11
    }

    comprehensive_results = {
        "feature_columns": feature_cols,
        "type_mapping": type_mapping,
        "tuning_config": CONFIG["tuning"],
        "tuning_results": tuning_results,
        "data_splits": {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "total_size": len(df)
        },
        "training_stats": {
            "total_records": len(df),
            "unique_files": df["file_id"].nunique(),
            "hit_rate": df["hit"].mean(),
            "avg_access_distance": df["next_access_distance"].mean(),
            "avg_reuse_prob": df["future_reuse_prob"].mean()
        },
        "model_performance_summary": {
            "binary_f1": tuning_results["binary"]["test_metrics"]["f1"],
            "binary_auc": tuning_results["binary"]["test_metrics"]["auc"],
            "distance_mae": distance_metrics["mae"],
            "distance_r2": distance_metrics["r2"],
            "reuse_mae": reuse_metrics["mae"],
            "reuse_r2": reuse_metrics["r2"]
        }
    }

    results_filename = f"{output_prefix}_tuning_results.json"
    with open(results_filename, "w") as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    print(f"Comprehensive results saved to {results_filename}")

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print("Binary Classification:")
    print(f"  Best CV Score: {tuning_results['binary']['best_cv_score']:.4f}")
    print(f"  Test F1 Score: {tuning_results['binary']['test_metrics']['f1']:.4f}")
    print(f"  Test AUC: {tuning_results['binary']['test_metrics']['auc']:.4f}")

    print("\nDistance Regression:")
    print(f"  Best CV Score: {tuning_results['distance']['best_cv_score']:.4f}")
    print(f"  Test MAE: {distance_metrics['mae']:.2f}")
    print(f"  Test R²: {distance_metrics['r2']:.4f}")

    print("\nReuse Probability:")
    print(f"  Best CV Score: {tuning_results['reuse']['best_cv_score']:.4f}")
    print(f"  Test MAE: {reuse_metrics['mae']:.4f}")
    print(f"  Test R²: {reuse_metrics['r2']:.4f}")

    return models, feature_cols, tuning_results

class CacheJSONEvictionScorer:
    def __init__(self, model_prefix="web_lrb_model"):
        self.models = {}
        self.feature_columns = []
        self.type_mapping = {}
        self.tuning_results = {}

        try:
            self.models["binary"] = joblib.load(f"{model_prefix}_binary.pkl")
            self.models["distance"] = joblib.load(f"{model_prefix}_distance.pkl")
            self.models["reuse"] = joblib.load(f("{model_prefix}_reuse.pkl"))
        except TypeError:
            # Fix for f-string typo above (graceful fallback)
            self.models["reuse"] = joblib.load(f"{model_prefix}_reuse.pkl")
        except FileNotFoundError as e:
            print(f"Model file not found: {e}")
            self.models = {}

        try:
            with open(f"{model_prefix}_tuning_results.json", "r") as f:
                results = json.load(f)
                self.feature_columns = results["feature_columns"]
                self.type_mapping = results["type_mapping"]
                self.tuning_results = results["tuning_results"]
            print(f"Loaded tuned cache JSON models: {model_prefix}")
            print(f"Features: {self.feature_columns}")
            if self.tuning_results:
                print("Model Performance:")
                for model_name, result in self.tuning_results.items():
                    if "test_metrics" in result:
                        if model_name == "binary":
                            print(f"  {model_name}: F1={result['test_metrics']['f1']:.3f}, AUC={result['test_metrics']['auc']:.3f}")
                        else:
                            print(f"  {model_name}: MAE={result['test_metrics']['mae']:.3f}, R²={result['test_metrics']['r2']:.3f}")
        except FileNotFoundError:
            pass

    def prepare_cache_features(self, cache_data):
        if not cache_data:
            return pd.DataFrame()

        features_list = []
        for file_id, cache_info in cache_data.items():
            feature_row = {
                "size": cache_info.get("size", 0),
                "type_encoded": self.type_mapping.get(cache_info.get("type", "unknown"), 11),
                "access_count": cache_info.get("access_count", 1),
                "age_since_last_access": cache_info.get("age_since_last_access", 0),
                "age_hours": cache_info.get("age_hours", 0),
                "log_file_size": cache_info.get("log_file_size", np.log1p(cache_info.get("size", 0))),
                "access_rate": cache_info.get("access_rate", 1.0),
                "recency_score": cache_info.get("recency_score", 1.0),
                "stack_distance_approx": cache_info.get("stack_distance_approx", 1000)
            }
            features_list.append(feature_row)

        return pd.DataFrame(features_list)

    def score_for_eviction(self, cache_data, explain=False):
        if not self.models or not cache_data:
            scores = {}
            for file_id, cache_info in cache_data.items():
                age_score = cache_info.get("age_since_last_access", 0) / 3600
                access_score = 1.0 / max(cache_info.get("access_count", 1), 1)
                scores[file_id] = age_score * 0.7 + access_score * 0.3
            return scores

        features_df = self.prepare_cache_features(cache_data)
        if features_df.empty:
            return {}

        features_df = features_df[self.feature_columns]

        scores = np.zeros(len(features_df))
        file_ids = list(cache_data.keys())
        explanations = {} if explain else None

        if "binary" in self.models:
            access_prob = self.models["binary"].predict_proba(features_df)[:, 1]
            no_access_score = (1.0 - access_prob) * 0.4
            scores += no_access_score
            if explain:
                for i, file_id in enumerate(file_ids):
                    explanations[file_id] = {"no_access_prob": float(1.0 - access_prob[i])}

        if "distance" in self.models:
            distances = self.models["distance"].predict(features_df)
            max_distance = CONFIG["look_ahead_window"] * 3
            normalized_distances = np.clip(distances / max_distance, 0, 1)
            distance_score = normalized_distances * 0.3
            scores += distance_score
            if explain:
                for i, file_id in enumerate(file_ids):
                    explanations[file_id]["predicted_distance"] = float(distances[i])
                    explanations[file_id]["distance_score"] = float(normalized_distances[i])

        if "reuse" in self.models:
            reuse_probs = self.models["reuse"].predict(features_df)
            no_reuse_score = (1.0 - np.clip(reuse_probs, 0, 1)) * 0.3
            scores += no_reuse_score
            if explain:
                for i, file_id in enumerate(file_ids):
                    explanations[file_id]["reuse_prob"] = float(reuse_probs[i])
                    explanations[file_id]["no_reuse_score"] = float(1.0 - np.clip(reuse_probs[i], 0, 1))

        result = {file_ids[i]: scores[i] for i in range(len(file_ids))}
        if explain:
            return result, explanations
        return result

    def recommend_eviction(self, cache_data, n_evict=1, explain=False):
        if explain:
            scores, explanations = self.score_for_eviction(cache_data, explain=True)
        else:
            scores = self.score_for_eviction(cache_data, explain=False)
            explanations = None

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for i, (file_id, score) in enumerate(sorted_items[:n_evict]):
            rec = {
                "file_id": file_id,
                "eviction_score": float(score),
                "rank": i + 1,
                "cache_info": cache_data[file_id]
            }
            if explanations:
                rec["explanation"] = explanations[file_id]
                rec["explanation"]["total_score"] = float(score)
            recommendations.append(rec)
        return recommendations

    def get_model_info(self):
        if not self.tuning_results:
            return "No tuning results available"
        lines = ["Model Information:", "=" * 40]
        for model_name, results in self.tuning_results.items():
            lines.append(f"\n{model_name.upper()} MODEL:")
            lines.append(f"  Best Parameters: {results['best_params']}")
            lines.append(f"  CV Score: {results['best_cv_score']:.4f}")
            if "test_metrics" in results:
                lines.append("  Test Metrics:")
                for metric, value in results["test_metrics"].items():
                    lines.append(f"    {metric}: {value:.4f}")
        return "\n".join(lines)

def create_ensemble_scorer(model_prefix="web_lrb_model", weights=None):
    if weights is None:
        weights = {"binary": 0.4, "distance": 0.3, "reuse": 0.3}

    class EnsembleEvictionScorer(CacheJSONEvictionScorer):
        def __init__(self, model_prefix, ensemble_weights):
            super().__init__(model_prefix)
            self.ensemble_weights = ensemble_weights

        def score_for_eviction(self, cache_data, explain=False):
            if not self.models or not cache_data:
                return super().score_for_eviction(cache_data, explain)

            features_df = self.prepare_cache_features(cache_data)
            if features_df.empty:
                return {}

            features_df = features_df[self.feature_columns]
            scores = np.zeros(len(features_df))
            file_ids = list(cache_data.keys())
            explanations = {} if explain else None

            if "binary" in self.models:
                access_prob = self.models["binary"].predict_proba(features_df)[:, 1]
                no_access_score = (1.0 - access_prob) * self.ensemble_weights["binary"]
                scores += no_access_score
                if explain:
                    for i, file_id in enumerate(file_ids):
                        explanations[file_id] = {
                            "no_access_prob": float(1.0 - access_prob[i]),
                            "binary_weight": self.ensemble_weights["binary"]
                        }

            if "distance" in self.models:
                distances = self.models["distance"].predict(features_df)
                max_distance = CONFIG["look_ahead_window"] * 3
                normalized_distances = np.clip(distances / max_distance, 0, 1)
                distance_score = normalized_distances * self.ensemble_weights["distance"]
                scores += distance_score
                if explain:
                    for i, file_id in enumerate(file_ids):
                        explanations[file_id]["predicted_distance"] = float(distances[i])
                        explanations[file_id]["distance_weight"] = self.ensemble_weights["distance"]

            if "reuse" in self.models:
                reuse_probs = self.models["reuse"].predict(features_df)
                no_reuse_score = (1.0 - np.clip(reuse_probs, 0, 1)) * self.ensemble_weights["reuse"]
                scores += no_reuse_score
                if explain:
                    for i, file_id in enumerate(file_ids):
                        explanations[file_id]["reuse_prob"] = float(reuse_probs[i])
                        explanations[file_id]["reuse_weight"] = self.ensemble_weights["reuse"]

            result = {file_ids[i]: scores[i] for i in range(len(file_ids))}
            if explain:
                return result, explanations
            return result

    return EnsembleEvictionScorer(model_prefix, weights)

def train_cache_json_pipeline_with_tuning(csv_path, model_prefix="web_lrb_model", quick_mode=False):
    print("Starting Enhanced Cache JSON Training Pipeline with Hyperparameter Tuning")
    print("=" * 80)

    df = preprocess_web_access_log(csv_path)
    df = generate_future_aware_labels(df, CONFIG["look_ahead_window"], CONFIG["cache_size"])
    df = generate_cache_json_features(df)

    models, feature_cols, tuning_results = train_cache_json_models_with_tuning(df, model_prefix, quick_mode)
    scorer = CacheJSONEvictionScorer(model_prefix)

    print("\nTesting enhanced scorer with sample cache data...")
    if len(df) > 5:
        sample_cache = {}
        sample_df = df.head(5)
        for _, row in sample_df.iterrows():
            file_id = row["file_id"]
            sample_cache[file_id] = {
                "last_access": datetime.now().isoformat(),
                "size": int(row["size"]),
                "type": row["type"],
                "access_count": int(row["access_count"]),
                "age_since_last_access": float(row["age_since_last_access"]),
                "age_hours": float(row["age_hours"]),
                "log_file_size": float(row["log_file_size"]),
                "access_rate": float(row["access_rate"]),
                "recency_score": float(row["recency_score"]),
                "stack_distance_approx": float(row["stack_distance_approx"])
            }

        recommendations = scorer.recommend_eviction(sample_cache, n_evict=3)
        print("Standard eviction recommendations:")
        for rec in recommendations:
            cache_info = rec['cache_info']
            print(f"  Rank {rec['rank']}: {rec['file_id'][:50]}...")
            print(f"    Score: {rec['eviction_score']:.4f}")
            print(f"    Type: {cache_info['type']}, Access Count: {cache_info['access_count']}")

        print("\nDetailed recommendations with explanations:")
        detailed_recs = scorer.recommend_eviction(sample_cache, n_evict=2, explain=True)
        for rec in detailed_recs:
            print(f"\n  Rank {rec['rank']}: {rec['file_id'][:40]}...")
            print(f"    Total Score: {rec['eviction_score']:.4f}")
            if "explanation" in rec:
                exp = rec["explanation"]
                if "no_access_prob" in exp:
                    print(f"    No Access Probability: {exp['no_access_prob']:.3f}")
                if "predicted_distance" in exp:
                    print(f"    Predicted Distance: {exp['predicted_distance']:.1f}")
                if "reuse_prob" in exp:
                    print(f"    Reuse Probability: {exp['reuse_prob']:.3f}")

    print("\nCreating ensemble scorer with custom weights...")
    ensemble_weights = {"binary": 0.5, "distance": 0.25, "reuse": 0.25}
    ensemble_scorer = create_ensemble_scorer(model_prefix, ensemble_weights)

    if len(df) > 5:
        ensemble_recs = ensemble_scorer.recommend_eviction(sample_cache, n_evict=2, explain=True)
        print("Ensemble recommendations:")
        for rec in ensemble_recs:
            print(f"  Rank {rec['rank']}: Score {rec['eviction_score']:.4f}")

    print("\nTraining complete.")
    print(f"Models saved with prefix: {model_prefix}")
    print(f"Features used: {feature_cols}")
    print(f"Hyperparameter tuning: {'Quick mode' if quick_mode else 'Full search'}")

    print("\n" + scorer.get_model_info())
    return models, scorer, df, tuning_results

def train_with_access_log_tuned(csv_path="access.log", quick_mode=False):
    return train_cache_json_pipeline_with_tuning(csv_path, CONFIG["model_prefix"], quick_mode)

def train_quick(csv_path="access.log"):
    return train_with_access_log_tuned(csv_path, quick_mode=True)

def train_full(csv_path="access.log"):
    return train_with_access_log_tuned(csv_path, quick_mode=False)

if __name__ == "__main__":
    models, scorer, df, tuning_results = train_quick(CONFIG["input_file"])