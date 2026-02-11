from openai import OpenAI
import pandas as pd
import json
from typing import Dict, List, Tuple
from app.config import settings
from app.core.logger import logger


class SignatureBuilder:
    """
    Generates table signatures and embeddings for similarity-based table matching.
    
    A signature includes:
    - Table name
    - Column names and types
    - Sample rows (first 5 rows)
    - Row count
    
    The signature is converted to a text representation and embedded using OpenAI's
    text-embedding-3-small model for similarity search.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.embedding_model
        self.embedding_dimensions = settings.embedding_dimensions
        logger.info(f"SignatureBuilder initialized with model: {self.embedding_model}")
    
    def generate_signature(
        self,
        df: pd.DataFrame,
        table_name: str,
        column_types: Dict[str, str]
    ) -> Dict:
        """
        Generate a table signature from DataFrame metadata.
        
        Args:
            df: DataFrame to generate signature from
            table_name: Proposed table name
            column_types: Dictionary mapping column names to PostgreSQL types
            
        Returns:
            Dictionary containing signature components
        """
        logger.info(f"Generating signature for table: {table_name}")
        
        # Get column names
        columns = df.columns.tolist()
        
        # Get sample rows (first 5)
        sample_rows = df.head(5).to_dict(orient='records')
        
        # Convert sample rows to JSON-serializable format
        sample_rows_serializable = []
        for row in sample_rows:
            serializable_row = {}
            for key, value in row.items():
                # Convert pandas/numpy types to Python native types
                if pd.isna(value):
                    serializable_row[key] = None
                elif isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)):
                    serializable_row[key] = str(value)
                elif hasattr(value, 'item'):  # numpy types
                    serializable_row[key] = value.item()
                else:
                    serializable_row[key] = value
            sample_rows_serializable.append(serializable_row)
        
        # Build signature
        signature = {
            "table_name": table_name,
            "columns": columns,
            "column_types": column_types,
            "sample_rows": sample_rows_serializable,
            "row_count": len(df)
        }
        
        logger.info(f"Signature generated: {len(columns)} columns, {len(sample_rows_serializable)} sample rows")
        return signature
    
    def create_embedding(self, signature: Dict) -> List[float]:
        """
        Create an embedding vector from a signature using OpenAI's embedding API.
        
        Args:
            signature: Signature dictionary
            
        Returns:
            Embedding vector (list of floats)
        """
        logger.info("Creating embedding from signature")
        
        try:
            # Convert signature to text representation
            # Focus on structure: table name, columns, types, and sample data patterns
            signature_text = self._signature_to_text(signature)
            
            logger.debug(f"Signature text (first 500 chars): {signature_text[:500]}")
            
            # Call OpenAI embeddings API
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=signature_text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.info(f"Embedding created: {len(embedding)} dimensions")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise
    
    def _signature_to_text(self, signature: Dict) -> str:
        """
        Convert signature dictionary to text representation for embedding.
        
        The text format emphasizes:
        - Table name and purpose
        - Column structure (names and types)
        - Sample data patterns
        
        Args:
            signature: Signature dictionary
            
        Returns:
            Text representation of signature
        """
        parts = []
        
        # Table name
        parts.append(f"Table: {signature['table_name']}")
        
        # Column structure
        parts.append(f"Columns ({len(signature['columns'])}):")
        for col in signature['columns']:
            col_type = signature['column_types'].get(col, 'UNKNOWN')
            parts.append(f"  - {col}: {col_type}")
        
        # Sample data patterns
        parts.append(f"Sample Data ({len(signature['sample_rows'])} rows):")
        for i, row in enumerate(signature['sample_rows'], 1):
            # Create a compact representation of the row
            row_str = ", ".join([f"{k}={v}" for k, v in list(row.items())[:5]])  # First 5 columns
            parts.append(f"  Row {i}: {row_str}")
        
        # Row count
        parts.append(f"Total Rows: {signature['row_count']}")
        
        return "\n".join(parts)
    
    def build_signature_with_embedding(
        self,
        df: pd.DataFrame,
        table_name: str,
        column_types: Dict[str, str]
    ) -> Tuple[Dict, List[float]]:
        """
        Generate signature and embedding in one call.
        
        Args:
            df: DataFrame to generate signature from
            table_name: Proposed table name
            column_types: Dictionary mapping column names to PostgreSQL types
            
        Returns:
            Tuple of (signature_dict, embedding_vector)
        """
        logger.info(f"Building signature with embedding for: {table_name}")
        
        # Generate signature
        signature = self.generate_signature(df, table_name, column_types)
        
        # Create embedding
        embedding = self.create_embedding(signature)
        
        logger.info("Signature and embedding built successfully")
        return signature, embedding


# Global signature builder instance
signature_builder = SignatureBuilder()
