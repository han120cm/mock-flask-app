# Cache Eviction Policies for CDN Testbed
from .base import CachePolicy
from .lru import LRUPolicy
from .lfu import LFUPolicy
from .fifo import FIFOPolicy
from .size_aware import SizeAwarePolicy
from .lrb import LRBPolicy

__all__ = [
    'CachePolicy',
    'LRUPolicy', 
    'LFUPolicy',
    'FIFOPolicy',
    'SizeAwarePolicy',
    'LRBPolicy'
]

AVAILABLE_POLICIES = {
    'lru': LRUPolicy,
    'lfu': LFUPolicy,
    'fifo': FIFOPolicy,
    'size': SizeAwarePolicy,
    'lrb': LRBPolicy
}

def get_policy(policy_name: str, **kwargs) -> CachePolicy:
    """Factory function to get a cache policy instance"""
    if policy_name not in AVAILABLE_POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(AVAILABLE_POLICIES.keys())}")
    return AVAILABLE_POLICIES[policy_name](**kwargs)
