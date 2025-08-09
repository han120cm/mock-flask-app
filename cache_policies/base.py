"""Base Cache Policy Interface for CDN Testbed"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd


class CachePolicy(ABC):
    """Abstract base class for cache eviction policies"""
    
    def __init__(self, cache_size_mb: float = 100):
        """
        Initialize cache policy
        
        Args:
            cache_size_mb: Maximum cache size in megabytes
        """
        self.cache_size_mb = cache_size_mb
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self.eviction_history = []
        self.policy_name = self.__class__.__name__
        
    @abstractmethod
    def score_items(self, cache_index: Dict[str, Any]) -> pd.DataFrame:
        """
        Score cache items for eviction. Higher scores = evict first.
        
        Args:
            cache_index: Dictionary with cache metadata
                {
                    "file_id": {
                        "size": int,  # bytes
                        "last_access": str,  # ISO timestamp
                        "creation_time": str,  # ISO timestamp
                        "access_count": int,
                        "type": str  # "image", "video", "other"
                    }
                }
        
        Returns:
            DataFrame with columns: file_id, eviction_score, size, metadata...
            Sorted by eviction_score descending (highest = evict first)
        """
        pass
    
    def select_victims(self, 
                       cache_index: Dict[str, Any], 
                       target_bytes: int = None,
                       dry_run: bool = True) -> Tuple[List[str], Dict[str, Any]]:
        """
        Select items to evict based on policy
        
        Args:
            cache_index: Cache metadata dictionary
            target_bytes: Bytes to free up. If None, uses cache_size_bytes limit
            dry_run: If True, only simulate eviction
            
        Returns:
            Tuple of (list of file_ids to evict, metrics dict)
        """
        # Score all items
        df = self.score_items(cache_index)
        
        # Calculate current cache usage
        current_size = df['size'].sum()
        
        # Determine eviction target
        if target_bytes is None:
            if current_size <= self.cache_size_bytes:
                return [], {
                    'current_size_mb': current_size / (1024 * 1024),
                    'limit_mb': self.cache_size_mb,
                    'evicted_count': 0,
                    'evicted_bytes': 0,
                    'policy': self.policy_name
                }
            target_bytes = current_size - self.cache_size_bytes
        
        # Select victims
        victims = []
        bytes_freed = 0
        
        for _, row in df.iterrows():
            if bytes_freed >= target_bytes:
                break
            victims.append(row['file_id'])
            bytes_freed += row['size']
        
        # Record metrics
        metrics = {
            'policy': self.policy_name,
            'current_size_mb': current_size / (1024 * 1024),
            'limit_mb': self.cache_size_mb,
            'target_bytes': target_bytes,
            'evicted_count': len(victims),
            'evicted_bytes': bytes_freed,
            'evicted_mb': bytes_freed / (1024 * 1024),
            'dry_run': dry_run,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if not dry_run:
            self.eviction_history.append(metrics)
        
        return victims, metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get policy statistics"""
        return {
            'policy_name': self.policy_name,
            'cache_size_mb': self.cache_size_mb,
            'eviction_count': len(self.eviction_history),
            'total_evicted_mb': sum(h['evicted_mb'] for h in self.eviction_history),
            'history': self.eviction_history
        }
