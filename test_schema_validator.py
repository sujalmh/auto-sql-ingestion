"""
Test script for schema validator.

This script tests:
1. Fetching table metadata from tables_metadata
2. Schema validation (exact match, missing columns, extra columns)
3. Discrepancy report generation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.schema_validator import schema_validator
from app.core.logger import logger


def test_fetch_metadata():
    """Test fetching table metadata."""
    print("\n" + "="*80)
    print("TEST 1: Fetch Table Metadata")
    print("="*80)
    
    # Test with a table that should exist (replace with your actual table name)
    table_name = input("Enter an existing table name to test (or press Enter to skip): ").strip()
    
    if not table_name:
        print("⊘ Skipped - no table name provided")
        return None
    
    try:
        metadata = schema_validator.fetch_table_metadata(table_name)
        
        if metadata:
            print(f"✓ Metadata fetched successfully")
            print(f"  - Table: {metadata['table_name']}")
            print(f"  - Columns ({len(metadata['columns'])}): {', '.join(metadata['columns'][:5])}...")
            print(f"  - Domain: {metadata['data_domain']}")
            print(f"  - Row count: {metadata['rows_count']}")
            return metadata
        else:
            print(f"✗ Table not found: {table_name}")
            return None
    except Exception as e:
        print(f"✗ Error fetching metadata: {e}")
        return None


def test_exact_match():
    """Test schema validation with exact match."""
    print("\n" + "="*80)
    print("TEST 2: Schema Validation - Exact Match")
    print("="*80)
    
    existing_columns = ['state', 'period', 'value', 'category']
    new_columns = ['state', 'period', 'value', 'category']
    
    try:
        result = schema_validator.validate_schema_match(new_columns, existing_columns)
        
        print(f"✓ Validation completed")
        print(f"  - Compatible: {result['is_compatible']}")
        print(f"  - Match percentage: {result['match_percentage']}%")
        print(f"  - Matching columns: {len(result['matching_columns'])}")
        print(f"  - Missing columns: {len(result['missing_columns'])}")
        print(f"  - Extra columns: {len(result['extra_columns'])}")
        
        assert result['is_compatible'] == True, "Should be compatible"
        assert result['match_percentage'] == 100.0, "Should be 100% match"
        print("✓ Test passed - exact match detected correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")


def test_missing_columns():
    """Test schema validation with missing columns."""
    print("\n" + "="*80)
    print("TEST 3: Schema Validation - Missing Columns")
    print("="*80)
    
    existing_columns = ['state', 'period', 'value', 'category', 'region']
    new_columns = ['state', 'period', 'value']  # Missing 'category' and 'region'
    
    try:
        result = schema_validator.validate_schema_match(new_columns, existing_columns)
        
        print(f"✓ Validation completed")
        print(f"  - Compatible: {result['is_compatible']}")
        print(f"  - Match percentage: {result['match_percentage']}%")
        print(f"  - Missing columns: {result['missing_columns']}")
        
        assert result['is_compatible'] == False, "Should be incompatible"
        assert 'category' in result['missing_columns'], "Should detect missing 'category'"
        assert 'region' in result['missing_columns'], "Should detect missing 'region'"
        print("✓ Test passed - missing columns detected correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")


def test_extra_columns():
    """Test schema validation with extra columns."""
    print("\n" + "="*80)
    print("TEST 4: Schema Validation - Extra Columns")
    print("="*80)
    
    existing_columns = ['state', 'period', 'value']
    new_columns = ['state', 'period', 'value', 'category', 'region']  # Extra columns
    
    try:
        result = schema_validator.validate_schema_match(new_columns, existing_columns)
        
        print(f"✓ Validation completed")
        print(f"  - Compatible: {result['is_compatible']}")
        print(f"  - Match percentage: {result['match_percentage']}%")
        print(f"  - Extra columns: {result['extra_columns']}")
        
        assert result['is_compatible'] == False, "Should be incompatible"
        assert 'category' in result['extra_columns'], "Should detect extra 'category'"
        assert 'region' in result['extra_columns'], "Should detect extra 'region'"
        print("✓ Test passed - extra columns detected correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")


def test_discrepancy_report():
    """Test discrepancy report generation."""
    print("\n" + "="*80)
    print("TEST 5: Discrepancy Report Generation")
    print("="*80)
    
    existing_columns = ['state', 'period', 'value', 'category']
    new_columns = ['state', 'period', 'amount', 'region']  # Different columns
    
    try:
        result = schema_validator.validate_schema_match(new_columns, existing_columns)
        report = schema_validator.generate_discrepancy_report(
            result,
            table_name="test_table",
            new_file_name="test_file.xlsx"
        )
        
        print("✓ Report generated:")
        print(report)
        
    except Exception as e:
        print(f"✗ Test failed: {e}")


def test_complete_workflow():
    """Test complete validation workflow."""
    print("\n" + "="*80)
    print("TEST 6: Complete Validation Workflow")
    print("="*80)
    
    table_name = input("Enter an existing table name for workflow test (or press Enter to skip): ").strip()
    
    if not table_name:
        print("⊘ Skipped - no table name provided")
        return
    
    # Simulate new file columns (modify as needed)
    new_columns = ['state', 'period', 'value']
    
    try:
        is_compatible, validation_result, report = schema_validator.validate_incremental_load(
            table_name=table_name,
            new_columns=new_columns,
            new_file_name="test_upload.xlsx"
        )
        
        print(f"✓ Workflow completed")
        print(f"  - Compatible: {is_compatible}")
        print(f"\nReport:")
        print(report)
        
    except Exception as e:
        print(f"✗ Workflow failed: {e}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("SCHEMA VALIDATOR TESTS")
    print("="*80)
    
    # Test 1: Fetch metadata (optional - requires existing table)
    metadata = test_fetch_metadata()
    
    # Test 2: Exact match
    test_exact_match()
    
    # Test 3: Missing columns
    test_missing_columns()
    
    # Test 4: Extra columns
    test_extra_columns()
    
    # Test 5: Discrepancy report
    test_discrepancy_report()
    
    # Test 6: Complete workflow (optional - requires existing table)
    test_complete_workflow()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
