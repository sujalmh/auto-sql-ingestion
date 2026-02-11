"""
Test script for signature builder and Milvus integration.

This script tests:
1. Signature generation from sample DataFrame
2. Embedding creation via OpenAI
3. Milvus connection and collection creation
4. Signature insertion and similarity search
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.signature_builder import signature_builder
from app.core.milvus_manager import milvus_manager
from app.core.logger import logger


def create_sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    data = {
        'state': ['Maharashtra', 'Gujarat', 'Karnataka', 'Tamil Nadu', 'West Bengal'],
        'period': pd.date_range('2023-01-01', periods=5, freq='M'),
        'value': [125.5, 130.2, 128.7, 135.1, 122.3],
        'category': ['Manufacturing', 'Manufacturing', 'Services', 'Services', 'Manufacturing']
    }
    return pd.DataFrame(data)


def test_signature_generation():
    """Test signature generation."""
    print("\n" + "="*80)
    print("TEST 1: Signature Generation")
    print("="*80)
    
    df = create_sample_dataframe()
    table_name = "iip_india_mth_sctg"
    column_types = {
        'state': 'VARCHAR(200)',
        'period': 'DATE',
        'value': 'NUMERIC(15,2)',
        'category': 'VARCHAR(100)'
    }
    
    try:
        signature = signature_builder.generate_signature(df, table_name, column_types)
        print(f"✓ Signature generated successfully")
        print(f"  - Table: {signature['table_name']}")
        print(f"  - Columns: {len(signature['columns'])}")
        print(f"  - Sample rows: {len(signature['sample_rows'])}")
        print(f"  - Row count: {signature['row_count']}")
        return signature
    except Exception as e:
        print(f"✗ Signature generation failed: {e}")
        return None


def test_embedding_creation(signature):
    """Test embedding creation."""
    print("\n" + "="*80)
    print("TEST 2: Embedding Creation")
    print("="*80)
    
    if not signature:
        print("✗ Skipping (no signature)")
        return None
    
    try:
        embedding = signature_builder.create_embedding(signature)
        print(f"✓ Embedding created successfully")
        print(f"  - Dimensions: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        return embedding
    except Exception as e:
        print(f"✗ Embedding creation failed: {e}")
        return None


def test_milvus_connection():
    """Test Milvus connection."""
    print("\n" + "="*80)
    print("TEST 3: Milvus Connection")
    print("="*80)
    
    try:
        success = milvus_manager.connect()
        if success:
            print("✓ Connected to Milvus successfully")
            return True
        else:
            print("✗ Failed to connect to Milvus")
            return False
    except Exception as e:
        print(f"✗ Milvus connection failed: {e}")
        return False


def test_collection_creation():
    """Test collection creation."""
    print("\n" + "="*80)
    print("TEST 4: Collection Creation")
    print("="*80)
    
    try:
        success = milvus_manager.create_collection()
        if success:
            print("✓ Collection created/loaded successfully")
            
            # Get stats
            stats = milvus_manager.get_collection_stats()
            print(f"  - Collection: {stats.get('collection_name')}")
            print(f"  - Total signatures: {stats.get('total_signatures')}")
            print(f"  - Embedding dimensions: {stats.get('embedding_dimensions')}")
            return True
        else:
            print("✗ Failed to create collection")
            return False
    except Exception as e:
        print(f"✗ Collection creation failed: {e}")
        return False


def test_signature_insertion(signature, embedding):
    """Test signature insertion."""
    print("\n" + "="*80)
    print("TEST 5: Signature Insertion")
    print("="*80)
    
    if not signature or not embedding:
        print("✗ Skipping (no signature or embedding)")
        return False
    
    try:
        success = milvus_manager.insert_signature(
            table_name=signature['table_name'],
            embedding=embedding,
            signature=signature
        )
        if success:
            print(f"✓ Signature inserted successfully for: {signature['table_name']}")
            return True
        else:
            print("✗ Failed to insert signature")
            return False
    except Exception as e:
        print(f"✗ Signature insertion failed: {e}")
        return False


def test_similarity_search(embedding):
    """Test similarity search."""
    print("\n" + "="*80)
    print("TEST 6: Similarity Search")
    print("="*80)
    
    if not embedding:
        print("✗ Skipping (no embedding)")
        return
    
    try:
        matches = milvus_manager.search_similar(embedding, top_k=5, threshold=0.0)
        print(f"✓ Similarity search completed")
        print(f"  - Found {len(matches)} matches")
        
        for i, match in enumerate(matches, 1):
            print(f"\n  Match {i}:")
            print(f"    - Table: {match['table_name']}")
            print(f"    - Similarity: {match['similarity_score']:.4f}")
            print(f"    - Created: {match['created_at']}")
    except Exception as e:
        print(f"✗ Similarity search failed: {e}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("SIGNATURE BUILDER & MILVUS INTEGRATION TESTS")
    print("="*80)
    
    # Test 1: Signature generation
    signature = test_signature_generation()
    
    # Test 2: Embedding creation
    embedding = test_embedding_creation(signature)
    
    # Test 3: Milvus connection
    connected = test_milvus_connection()
    
    if not connected:
        print("\n✗ Cannot proceed without Milvus connection")
        print("Please ensure:")
        print("  1. Milvus server is running")
        print("  2. MILVUS_HOST and MILVUS_PORT in .env are correct")
        print("  3. Authentication credentials (if required) are set")
        return
    
    # Test 4: Collection creation
    collection_ready = test_collection_creation()
    
    if not collection_ready:
        print("\n✗ Cannot proceed without collection")
        return
    
    # Test 5: Signature insertion
    inserted = test_signature_insertion(signature, embedding)
    
    # Test 6: Similarity search
    if inserted:
        test_similarity_search(embedding)
    
    # Cleanup
    print("\n" + "="*80)
    print("CLEANUP")
    print("="*80)
    milvus_manager.disconnect()
    print("✓ Disconnected from Milvus")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
