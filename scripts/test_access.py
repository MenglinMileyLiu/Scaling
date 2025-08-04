#!/usr/bin/env python3
"""
Test Access to Political Text Scaling Dataset

Comprehensive testing script to verify dataset accessibility and functionality
using the consolidated data utilities.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scaling.data_utils import TextScalingDataset, load_dataset_splits, compute_topic_statistics


def test_basic_access():
    """Test basic dataset loading and access."""
    print("🔍 Testing Basic Dataset Access")
    print("-" * 40)
    
    try:
        # Load training split
        dataset = TextScalingDataset(
            source="mliliu/political-text-scaling",
            split="train",
            sample_size=100  # Small sample for testing
        )
        
        print(f"✅ Dataset loaded successfully")
        print(f"📊 Dataset size: {len(dataset):,} documents")
        print(f"🏷️  Topic columns: {len(dataset.topic_cols)}")
        print(f"📋 Available columns: {len(dataset.df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False


def test_data_structure():
    """Test data structure and content."""
    print("\n📋 Testing Data Structure")
    print("-" * 40)
    
    try:
        dataset = TextScalingDataset(
            source="mliliu/political-text-scaling",
            split="train",
            sample_size=50
        )
        
        # Test single document access
        doc = dataset[0]
        print(f"✅ Single document access works")
        print(f"📝 Document keys: {list(doc.keys())[:5]}... ({len(doc)} total)")
        
        # Test batch access
        batch = dataset.get_batch([0, 1, 2])
        print(f"✅ Batch access works: {batch.shape}")
        
        # Test sample
        sample = dataset.get_sample(3)
        print(f"✅ Random sampling works: {sample.shape}")
        
        # Show sample content
        sample_doc = sample.iloc[0]
        print(f"\n📖 Sample document:")
        print(f"   ID: {sample_doc.get('doc.id', 'N/A')}")
        print(f"   Date: {sample_doc.get('date', 'N/A')}")
        print(f"   Congress: {sample_doc.get('congno', 'N/A')}")
        print(f"   Body: {sample_doc.get('legis.body', 'N/A')}")
        print(f"   Text preview: {sample_doc.get('text', '')[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


def test_topic_analysis():
    """Test topic weight analysis."""
    print("\n🏷️  Testing Topic Analysis")
    print("-" * 40)
    
    try:
        dataset = TextScalingDataset(
            source="mliliu/political-text-scaling",
            split="train",
            sample_size=100
        )
        
        # Get topic statistics
        topic_stats = compute_topic_statistics(dataset)
        
        if topic_stats:
            print(f"✅ Topic analysis works")
            print(f"📊 Topic statistics computed for {len(dataset.topic_cols)} topics")
            
            # Show top dominant topics
            if 'dominant_topic_counts' in topic_stats:
                print(f"\n🥇 Top 5 dominant topics:")
                for i, (topic, count) in enumerate(list(topic_stats['dominant_topic_counts'].items())[:5]):
                    clean_name = topic.replace('tw.', '').replace('.', ' ').title()
                    print(f"   {i+1}. {clean_name}: {count} documents")
            
            # Show sample topic weights
            sample_doc = dataset.get_sample(1).iloc[0]
            topic_weights = {k: v for k, v in sample_doc.items() if k.startswith('tw.') and v > 0}
            sorted_weights = sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"\n📈 Sample document topic weights:")
            for topic, weight in sorted_weights:
                clean_name = topic.replace('tw.', '').replace('.', ' ').title()
                print(f"   {clean_name}: {weight:.4f}")
        else:
            print("⚠️  No topic statistics available")
        
        return True
        
    except Exception as e:
        print(f"❌ Topic analysis test failed: {e}")
        return False


def test_filtering():
    """Test dataset filtering capabilities."""
    print("\n🔍 Testing Dataset Filtering")
    print("-" * 40)
    
    try:
        # Test congress filter
        recent_dataset = TextScalingDataset(
            source="mliliu/political-text-scaling",
            split="train",
            filters={'congno': lambda x: x >= 110},
            sample_size=50
        )
        
        print(f"✅ Congress filtering works: {len(recent_dataset)} documents")
        
        # Test legislative body filter
        house_dataset = TextScalingDataset(
            source="mliliu/political-text-scaling",
            split="train",
            filters={'legis.body': lambda x: str(x) == 'US House'},
            sample_size=50
        )
        
        print(f"✅ Legislative body filtering works: {len(house_dataset)} documents")
        
        # Test column selection
        text_only_dataset = TextScalingDataset(
            source="mliliu/political-text-scaling",
            split="train",
            columns=['doc.id', 'text', 'date', 'congno'],
            sample_size=20
        )
        
        print(f"✅ Column selection works: {text_only_dataset.df.shape[1]} columns")
        
        return True
        
    except Exception as e:
        print(f"❌ Filtering test failed: {e}")
        return False


def test_batch_iteration():
    """Test batch iteration functionality."""
    print("\n⚡ Testing Batch Iteration")
    print("-" * 40)
    
    try:
        dataset = TextScalingDataset(
            source="mliliu/political-text-scaling",
            split="train",
            sample_size=100
        )
        
        batch_count = 0
        total_docs = 0
        
        for batch in dataset.iterate_batches(batch_size=25, shuffle=True):
            batch_count += 1
            total_docs += len(batch)
            if batch_count == 3:  # Test first 3 batches
                break
        
        print(f"✅ Batch iteration works")
        print(f"📦 Processed {batch_count} batches")
        print(f"📊 Total documents in batches: {total_docs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch iteration test failed: {e}")
        return False


def test_all_splits():
    """Test loading all dataset splits."""
    print("\n📂 Testing All Dataset Splits")
    print("-" * 40)
    
    try:
        train_ds, val_ds, test_ds = load_dataset_splits("mliliu/political-text-scaling")
        
        print(f"✅ All splits loaded successfully")
        print(f"📊 Train: {len(train_ds):,} documents")
        print(f"📊 Validation: {len(val_ds):,} documents") 
        print(f"📊 Test: {len(test_ds):,} documents")
        print(f"📊 Total: {len(train_ds) + len(val_ds) + len(test_ds):,} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Split loading test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Political Text Scaling Dataset - Comprehensive Tests")
    print("=" * 60)
    
    tests = [
        test_basic_access,
        test_data_structure,
        test_topic_analysis,
        test_filtering,
        test_batch_iteration,
        test_all_splits
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n🏁 Test Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed! Dataset is fully functional.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())