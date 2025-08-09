"""Size-Aware Cache Eviction Policy (GreedyDual-Size inspired)"""

from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import CachePolicy


class SizeAwarePolicy(CachePolicy):
    """
    Size-aware eviction policy - considers both size and value of items
    Inspired by GreedyDual-Size: prioritizes small, frequently accessed items
    """
    
    def score_items(self, cache_index: Dict[str, Any]) -> pd.DataFrame:
        """
        Score items by size-adjusted value (lower value per byte = evict first)
        Value = access_frequency / size
        """
        records = []
        now = datetime.utcnow()
        
        for file_id, metadata in cache_index.items():
            try:
                size = max(metadata.get('size', 1), 1)  # Avoid division by zero
                access_count = metadata.get('access_count', 0)
                last_access = datetime.fromisoformat(metadata['last_access'])
                age_seconds = (now - last_access).total_seconds()
                
                # Calculate value per byte
                # Higher access count = higher value
                # Larger size = lower value per byte
                # Recent access = bonus to value
                recency_factor = 1.0 / (1.0 + age_seconds / 3600)  # Decay over hours
                value_per_byte = (access_count + 1) * recency_factor / size
                
                # Eviction score: inverse of value (lower value = higher score = evict first)
                eviction_score = 1.0 / (value_per_byte + 0.0001)  # Small epsilon to avoid division by zero
                
                records.append({
                    'file_id': file_id,
                    'eviction_score': eviction_score,
                    'size': size,
                    'size_mb': size / (1024 * 1024),
                    'access_count': access_count,
                    'value_per_byte': value_per_byte,
                    'last_access': metadata['last_access'],
                    'age_hours': age_seconds / 3600,
                    'type': metadata.get('type', 'unknown')
                })
            except Exception as e:
                print(f"Error processing {file_id}: {e}")
                continue
        
        df = pd.DataFrame(records)
        if df.empty:
            return df
        
        # Sort by eviction score (descending - highest score evicted first)
        return df.sort_values('eviction_score', ascending=False)
