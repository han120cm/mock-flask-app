"""Least Frequently Used (LFU) Cache Eviction Policy"""

from datetime import datetime
import pandas as pd
from typing import Dict, Any
from .base import CachePolicy


class LFUPolicy(CachePolicy):
    """
    LFU eviction policy - evicts items with lowest access frequency
    """
    
    def score_items(self, cache_index: Dict[str, Any]) -> pd.DataFrame:
        """
        Score items by access frequency (lower frequency = higher score = evict first)
        Ties are broken by age (older items evicted first)
        """
        records = []
        now = datetime.utcnow()
        
        for file_id, metadata in cache_index.items():
            try:
                access_count = metadata.get('access_count', 0)
                last_access = datetime.fromisoformat(metadata['last_access'])
                age_seconds = (now - last_access).total_seconds()
                
                # Primary score: inverse of access count (lower count = higher score)
                # Secondary: age for tie-breaking
                # Score formula: (1 / (access_count + 1)) * 1000000 + age_seconds
                eviction_score = (1.0 / (access_count + 1)) * 1000000 + age_seconds
                
                records.append({
                    'file_id': file_id,
                    'eviction_score': eviction_score,
                    'size': metadata.get('size', 0),
                    'access_count': access_count,
                    'last_access': metadata['last_access'],
                    'age_seconds': age_seconds,
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
