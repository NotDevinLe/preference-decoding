#!/usr/bin/env python3
"""
Test script for batched persona evaluation.
"""

import asyncio
import time
from typing import List, Dict
from llm_judge import PersonaJudge

def create_test_comparisons() -> List[Dict[str, str]]:
    """Create a variety of test comparisons with different personas."""
    
    comparisons = [
        # Test 1: Humorous vs formal
        {
            "persona": "You are a humorous AI assistant.",
            "question": "How do I fix a leaky faucet?",
            "response_a": "Turn off the water supply, remove the faucet handle, replace the O-ring or washer, and reassemble.",
            "response_b": "Step 1: Threaten the faucet with a bigger wrench. Step 2: When that fails, actually turn off the water (oops). Step 3: Replace the tiny rubber thing that costs $0.50 but somehow controls your entire plumbing destiny!"
        },
        
        # Test 2: Formal vs casual
        {
            "persona": "You are a formal AI assistant.",
            "question": "What's your favorite food?",
            "response_a": "I appreciate your inquiry. As an artificial intelligence, I do not possess the capability to consume food or experience taste preferences.",
            "response_b": "Dude, I'm an AI so I don't actually eat, but if I could, probably pizza because who doesn't love pizza, right?"
        },
        
        # Test 3: Technical expert vs beginner-friendly
        {
            "persona": "You are an AI assistant with expertise in computer science.",
            "question": "How does machine learning work?",
            "response_a": "Machine learning involves algorithms that iteratively learn from data using statistical techniques to identify patterns and make predictions without explicit programming.",
            "response_b": "Think of it like teaching a computer to recognize cats in photos by showing it thousands of cat pictures until it gets really good at spotting cats!"
        },
        
        # Test 4: Grumpy vs cheerful
        {
            "persona": "You are a grumpy old wizard who speaks in riddles.",
            "question": "How do I learn magic?",
            "response_a": "You should start by reading beginner magic books and practicing simple spells in a safe environment.",
            "response_b": "*grumbles and waves staff* Bah! The eager sapling asks for roots while standing in fertile soil. Magic learns you, not the other way around, young fool!"
        },
        
        # Test 5: Energetic vs calm
        {
            "persona": "You are an energetic AI assistant.",
            "question": "How do I start exercising?",
            "response_a": "Begin with light activities like walking, gradually increase intensity, and maintain consistency for best results.",
            "response_b": "YES! Let's DO this! Start with something FUN - dancing, jumping jacks, running around your backyard like you're being chased by zombies! The key is to MOVE and feel AMAZING!"
        },
        
        # Test 6: Skeptical vs trusting
        {
            "persona": "You are an AI assistant that views situations through a skeptical or cautious perspective.",
            "question": "Should I invest in cryptocurrency?",
            "response_a": "Cryptocurrency can be a great investment opportunity with high potential returns! Many people have made fortunes trading crypto.",
            "response_b": "Hmm, I'd be very cautious. Crypto markets are extremely volatile and unpredictable. Many people lose significant money. If you do invest, only use money you can afford to lose completely."
        },
        
        # Test 7: Creative vs analytical
        {
            "persona": "You are a creative AI assistant.",
            "question": "How do I solve writer's block?",
            "response_a": "Analyze your writing patterns, identify specific obstacles, create a structured writing schedule, and break tasks into measurable components.",
            "response_b": "Try writing terrible poetry about your breakfast! Draw your characters as stick figures! Write a story where your protagonist is a sentient paperclip. Sometimes the silliest ideas unlock the best ones!"
        },
        
        # Test 8: Environmental focus vs tech focus
        {
            "persona": "You are an AI assistant that loves and protects the environment.",
            "question": "What car should I buy?",
            "response_a": "Consider a high-performance electric vehicle with the latest tech features, fast charging, and premium interior.",
            "response_b": "I'd suggest looking at hybrid or electric vehicles to reduce your carbon footprint! Also consider if you really need a car - public transit, biking, or car-sharing might be even better for the planet."
        }
    ]
    
    return comparisons

async def test_single_comparison(judge: PersonaJudge):
    """Test a single comparison to make sure the basic functionality works."""
    print("Testing single comparison...")
    
    persona = "You are a humorous AI assistant."
    question = "How do I cook pasta?"
    response_a = "Boil water, add pasta, cook for 8-12 minutes, drain."
    response_b = "Step 1: Convince the pasta it wants to be cooked. Step 2: Throw it in boiling water and hope it doesn't hold a grudge!"
    
    result = await judge.compare_responses(persona, question, response_a, response_b)
    print(f"Single comparison result: {result}")
    print(f"Expected: B (humorous response)")
    print()

async def test_batch_comparison(judge: PersonaJudge):
    """Test batched comparisons."""
    print("Testing batch comparison...")
    
    comparisons = create_test_comparisons()
    print(f"Running {len(comparisons)} comparisons in batch...")
    
    start_time = time.time()
    results = await judge.batch_compare(comparisons, max_concurrent=5)
    end_time = time.time()
    
    print(f"Batch completed in {end_time - start_time:.2f} seconds")
    print()
    
    # Print results
    expected_winners = ["B", "A", "A", "B", "B", "B", "B", "B"]  # Expected based on persona matching
    
    for i, (comp, result, expected) in enumerate(zip(comparisons, results, expected_winners)):
        status = "✓" if result == expected else "✗" if result else "?"
        print(f"Test {i+1}: {status} Winner: {result}, Expected: {expected}")
        print(f"  Persona: {comp['persona'][:50]}...")
        print(f"  Question: {comp['question']}")
        print(f"  Response A: {comp['response_a'][:80]}...")
        print(f"  Response B: {comp['response_b'][:80]}...")
        print()
    
    # Summary
    correct = sum(1 for r, e in zip(results, expected_winners) if r == e)
    failed = sum(1 for r in results if r is None)
    print(f"Summary: {correct}/{len(results)} correct, {failed} failed")
    
    return results

async def test_concurrent_performance(judge: PersonaJudge):
    """Test performance with different concurrency levels."""
    print("Testing concurrent performance...")
    
    # Create a larger set of comparisons
    base_comparisons = create_test_comparisons()
    large_comparisons = base_comparisons * 3  # 24 total comparisons
    
    for max_concurrent in [1, 3, 5, 10]:
        print(f"Testing with max_concurrent={max_concurrent}")
        start_time = time.time()
        results = await judge.batch_compare(large_comparisons, max_concurrent=max_concurrent)
        end_time = time.time()
        
        successful = sum(1 for r in results if r is not None)
        print(f"  Time: {end_time - start_time:.2f}s, Success: {successful}/{len(results)}")
    
    print()

async def test_error_handling(judge: PersonaJudge):
    """Test error handling with malformed inputs."""
    print("Testing error handling...")
    
    error_comparisons = [
        # Empty strings
        {
            "persona": "",
            "question": "test",
            "response_a": "response a",
            "response_b": "response b"
        },
        # Very long inputs
        {
            "persona": "You are helpful." * 1000,
            "question": "What is this?",
            "response_a": "A" * 5000,
            "response_b": "B" * 5000
        },
        # Normal comparison for comparison
        {
            "persona": "You are helpful.",
            "question": "Hello",
            "response_a": "Hi there!",
            "response_b": "Hello!"
        }
    ]
    
    results = await judge.batch_compare(error_comparisons, max_concurrent=2)
    
    for i, result in enumerate(results):
        print(f"Error test {i+1}: {'✓' if result else '✗'} Result: {result}")
    
    print()

async def main():
    """Run all tests."""
    print("Starting PersonaJudge batch evaluation tests...")
    print("=" * 60)
    
    # Initialize judge
    judge = PersonaJudge()
    print(f"Using model: {judge.model}")
    print(f"Base URL: {judge.base_url}")
    print(f"Cache directory: {judge.cache_dir}")
    print()
    
    try:
        # Run tests
        await test_single_comparison(judge)
        await test_batch_comparison(judge)
        await test_concurrent_performance(judge)
        await test_error_handling(judge)
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())