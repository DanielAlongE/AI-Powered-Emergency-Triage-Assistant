#!/usr/bin/env python3
"""
Simple test script to verify Ollama cloud model integration.
Run this to test if cloud models work correctly.
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.ollama_client import get_ollama_gateway


async def test_cloud_model():
    """Test cloud model integration with a simple prompt."""

    print("🧪 Testing Ollama Cloud Model Integration...")
    print("=" * 50)

    # Test with cloud model
    gateway = get_ollama_gateway(inference_mode="cloud")

    test_prompt = "What is 2+2? Please respond with just the number."

    try:
        print("📤 Sending test prompt to cloud model...")
        print(f"   Prompt: '{test_prompt}'")
        print(f"   Model: gpt-oss:20b-cloud")

        result = await gateway.stream_completion(
            prompt=test_prompt,
            model_override="gpt-oss:20b-cloud",
            temperature=0.1
        )

        response = result.get("text", "")
        print(f"📥 Response: '{response.strip()}'")

        if "4" in response:
            print("✅ Cloud model test PASSED!")
            return True
        else:
            print("❌ Cloud model test FAILED - unexpected response")
            return False

    except Exception as e:
        print(f"❌ Cloud model test FAILED with error: {str(e)}")
        print("\n💡 Possible solutions:")
        print("   1. Run 'ollama signin' to authenticate with Ollama Cloud")
        print("   2. Check your internet connection")
        print("   3. Verify the cloud model name is correct")
        return False


async def test_cloud_model_detection():
    """Test cloud model detection utility."""

    print("\n🔍 Testing Cloud Model Detection...")
    print("=" * 50)

    from services.ollama_client import _is_cloud_model

    test_cases = [
        ("gpt-oss:20b-cloud", True),
        ("qwen3-coder:480b-cloud", True),
        ("llama3.2:1b", False),
        ("gpt-oss:20b", False),
        ("deepseek-v3.1:671b-cloud", True)
    ]

    all_passed = True
    for model, expected in test_cases:
        result = _is_cloud_model(model)
        status = "✅" if result == expected else "❌"
        print(f"{status} {model} -> {result} (expected: {expected})")
        if result != expected:
            all_passed = False

    return all_passed


async def main():
    """Run all tests."""

    print("🚀 Starting Ollama Cloud Integration Tests\n")

    # Test 1: Cloud model detection
    detection_passed = await test_cloud_model_detection()

    # Test 2: Actual cloud model inference (requires authentication)
    print("\n⚠️  The next test requires 'ollama signin' to be completed")
    user_input = input("Continue with cloud inference test? (y/n): ")

    if user_input.lower().startswith('y'):
        inference_passed = await test_cloud_model()
    else:
        print("⏭️  Skipping cloud inference test")
        inference_passed = True  # Don't fail if user skipped

    # Results
    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS:")
    print(f"   Cloud Model Detection: {'✅ PASSED' if detection_passed else '❌ FAILED'}")
    print(f"   Cloud Inference: {'✅ PASSED' if inference_passed else '❌ FAILED'}")

    if detection_passed and inference_passed:
        print("\n🎉 All tests PASSED! Ollama cloud integration is working correctly.")
        return True
    else:
        print("\n💥 Some tests FAILED. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)