#!/usr/bin/env python3
"""
Quick test to verify package imports work correctly.
"""

def test_imports():
    """Test that all major imports work"""
    print("Testing package imports...")
    
    try:
        import gumbel
        print("✅ Main gumbel package imported")
    except ImportError as e:
        print(f"❌ Failed to import gumbel: {e}")
        return False
    
    try:
        from gumbel.utils import async_utils
        print("✅ async_utils imported")
    except ImportError as e:
        print(f"❌ Failed to import async_utils: {e}")
        return False
    
    try:
        from gumbel.core import DataSampler
        print("✅ DataSampler imported")
    except ImportError as e:
        print(f"❌ Failed to import DataSampler: {e}")
        return False
    
    try:
        from gumbel.utils import get_log_probs_async, build_full_prompt
        print("✅ Utility functions imported")
    except ImportError as e:
        print(f"❌ Failed to import utility functions: {e}")
        return False
        
    print("\n🎉 All imports successful! Package structure is working.")
    return True

if __name__ == "__main__":
    test_imports()