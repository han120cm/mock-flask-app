"""Least Recently Used (LRU) Cache Eviction Policy"""

from datetime import datetime
import pandas as pd
from typing import Dict, Any
from .base import CachePolicy


class LRUPolicy(CachePolicy):
    """
    LRU eviction policy - evicts items that haven't been accessed recently
    """
    
    def score_items(self, cache_index: Dict[str, Any]) -> pd.DataFrame:
        """
        Score items by age since last access (older = higher score = evict first)
        """
        records = []
        now = datetime.utcnow()
        
        for file_id, metadata in cache_index.items():
            try:
                last_access = datetime.fromisoformat(metadata['last_access'])
                age_seconds = (now - last_access).total_seconds()
                
                records.append({
                    'file_id': file_id,
                    'eviction_score': age_seconds,  # Older = higher score
                    'size': metadata.get('size', 0),
                    'last_access': metadata['last_access'],
                    'age_seconds': age_seconds,
                    'age_hours': age_seconds / 3600,
                    'access_count': metadata.get('access_count', 0),
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
