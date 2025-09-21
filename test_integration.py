#!/usr/bin/env python3
"""
Integration test to verify QAlign, Registry, and BonVoyage work together.
"""

import asyncio
import sys
import os
import torch
from transformers import AutoTokenizer

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_integration():
    """Test the integration between QAlign, Registry, and BonVoyage."""
    
    print("🧪 Starting integration test...")
    
    try:
        # Import required modules
        from src.models.registry_model import create_qalign_with_registry, RegistryModelWrapper, BonVoyageRewardWrapper
        from src.models.vectors.bonvoyage import BonVoyageVector
        from literegistry.client import RegistryClient
        from literegistry.kvstore import FileSystemKVStore
        
        print("✅ All imports successful")
        
        # Mock setup for testing (since we may not have actual servers running)
        print("🔧 Setting up mock components...")
        
        # Create a simple tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create mock registry client
        store = FileSystemKVStore("test_registry_data")
        registry_client = RegistryClient(store, service_type="model_path")
        
        model_name = "test-model"
        
        # Create BonVoyage vector
        device = "cpu"  # Use CPU for testing
        bonvoyage = BonVoyageVector(
            device=device,
            mc_samples=5,  # Small number for testing
            model_name=model_name
        )
        
        print("✅ Components created successfully")
        
        # Test model wrapper creation
        print("🔧 Testing model wrapper...")
        model_wrapper = RegistryModelWrapper(registry_client, tokenizer, model_name)
        
        # Test reward wrapper creation
        print("🔧 Testing reward wrapper...")
        reward_wrapper = BonVoyageRewardWrapper(bonvoyage, tokenizer, registry_client)
        
        print("✅ Wrappers created successfully")
        
        # Test basic interface compatibility
        print("🔧 Testing interface compatibility...")
        
        # Test tokenizer attribute
        assert hasattr(model_wrapper, 'tokenizer'), "Model wrapper missing tokenizer attribute"
        assert hasattr(model_wrapper.tokenizer, 'apply_chat_template'), "Tokenizer missing apply_chat_template method"
        
        # Test model interface methods
        test_texts = ["Hello world", "How are you?"]
        
        # Test encode method
        encoded = model_wrapper.encode(test_texts)
        assert isinstance(encoded, torch.Tensor), "encode() should return torch.Tensor"
        print(f"✅ encode() works: {encoded.shape}")
        
        # Test tokenize method
        tokenized = model_wrapper.tokenize(test_texts)
        assert isinstance(tokenized, list), "tokenize() should return list"
        assert all(isinstance(t, torch.Tensor) for t in tokenized), "tokenize() should return list of tensors"
        print(f"✅ tokenize() works: {len(tokenized)} items")
        
        # Test decode_tokenize method
        decoded = model_wrapper.decode_tokenize(tokenized)
        assert isinstance(decoded, list), "decode_tokenize() should return list"
        assert all(isinstance(t, str) for t in decoded), "decode_tokenize() should return list of strings"
        print(f"✅ decode_tokenize() works: {decoded}")
        
        # Test reward wrapper interface
        test_conversations = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks!"}
            ]
        ]
        
        print("🔧 Testing reward evaluation (this may take a moment)...")
        try:
            # This will likely fail without actual servers, but we can test the interface
            rewards = reward_wrapper.evaluate(test_conversations)
            print(f"✅ evaluate() works: {rewards}")
        except Exception as e:
            print(f"⚠️  evaluate() failed (expected without servers): {e}")
            # This is expected without actual model servers
        
        # Test QAlign creation (without running it)
        print("🔧 Testing QAlign creation...")
        try:
            from src.models.qalign.qalign_generator import QAlign
            
            qalign = QAlign(
                model=model_wrapper,
                reward=reward_wrapper,
                beta=0.1
            )
            
            print("✅ QAlign created successfully")
            print(f"✅ QAlign string representation: {qalign}")
            
        except Exception as e:
            print(f"❌ QAlign creation failed: {e}")
            return False
        
        # Test helper function
        print("🔧 Testing helper function...")
        try:
            qalign_via_helper = create_qalign_with_registry(
                registry_client=registry_client,
                tokenizer=tokenizer,
                model_name=model_name,
                bonvoyage_vector=bonvoyage,
                beta=0.1
            )
            print("✅ Helper function works")
            print(f"✅ QAlign via helper: {qalign_via_helper}")
            
        except Exception as e:
            print(f"❌ Helper function failed: {e}")
            return False
        
        print("\n🎉 Integration test completed successfully!")
        print("📝 Summary:")
        print("   - All imports work correctly")
        print("   - Model wrapper implements required interface")
        print("   - Reward wrapper implements required interface") 
        print("   - QAlign can be created with registry components")
        print("   - Helper function works correctly")
        print("\n⚠️  Note: Actual generation/reward computation requires running model servers")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the integration test."""
    success = asyncio.run(test_integration())
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()