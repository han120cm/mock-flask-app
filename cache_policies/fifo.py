"""First In First Out (FIFO) Cache Eviction Policy"""

from datetime import datetime
import pandas as pd
from typing import Dict, Any
from .base import CachePolicy


class FIFOPolicy(CachePolicy):
    """
    FIFO eviction policy - evicts oldest items first (by creation time)
    """
    
    def score_items(self, cache_index: Dict[str, Any]) -> pd.DataFrame:
        """
        Score items by creation time (older = higher score = evict first)
        """
        records = []
        now = datetime.utcnow()
        
        for file_id, metadata in cache_index.items():
            try:
                # Use creation_time if available, otherwise fall back to last_access
                creation_str = metadata.get('creation_time', metadata.get('last_access'))
                creation_time = datetime.fromisoformat(creation_str)
                age_seconds = (now - creation_time).total_seconds()
                
                records.append({
                    'file_id': file_id,
                    'eviction_score': age_seconds,  # Older = higher score
                    'size': metadata.get('size', 0),
                    'creation_time': creation_str,
                    'age_seconds': age_seconds,
                    'age_hours': age_seconds / 3600,
                    'age_days': age_seconds / 86400,
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
