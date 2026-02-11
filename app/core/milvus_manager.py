from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
from app.config import settings
from app.core.logger import logger


class MilvusManager:
    """
    Manages Milvus vector database operations for table signature storage and similarity search.
    
    Collection Schema:
    - id: INT64 (primary key, auto-increment)
    - table_name: VARCHAR (255)
    - embedding: FLOAT_VECTOR (1536 dimensions)
    - signature_json: VARCHAR (65535)
    - created_at: VARCHAR (50)
    """
    
    def __init__(self):
        self.collection_name = settings.milvus_collection
        self.embedding_dim = settings.embedding_dimensions
        self.collection: Optional[Collection] = None
        logger.info(f"MilvusManager initialized for collection: {self.collection_name}")
    
    def connect(self) -> bool:
        """
        Establish connection to Milvus server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to Milvus at {settings.milvus_host}:{settings.milvus_port}")
            
            # Build connection parameters
            conn_params = {
                "alias": "default",
                "host": settings.milvus_host,
                "port": settings.milvus_port
            }
            
            # Add authentication if provided
            if settings.milvus_user and settings.milvus_password:
                conn_params["user"] = settings.milvus_user
                conn_params["password"] = settings.milvus_password
                logger.info("Using authenticated connection")
            
            # Add database name (Milvus 2.x supports multiple databases)
            if settings.milvus_db:
                conn_params["db_name"] = settings.milvus_db
                logger.info(f"Connecting to database: {settings.milvus_db}")
            
            connections.connect(**conn_params)
            logger.info("Successfully connected to Milvus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from Milvus server."""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {str(e)}")
    
    def create_collection(self) -> bool:
        """
        Create the table_signatures collection if it doesn't exist.
        
        Returns:
            True if collection created or already exists, False on error
        """
        try:
            # Check if collection already exists
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                self.collection = Collection(self.collection_name)
                self.collection.load()
                return True
            
            logger.info(f"Creating collection: {self.collection_name}")
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="signature_json", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Table signatures for similarity-based matching"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            
            # Create index for vector similarity search
            index_params = {
                "metric_type": "COSINE",  # Cosine similarity
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info("Vector index created successfully")
            
            # Load collection into memory
            self.collection.load()
            logger.info("Collection loaded into memory")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return False
    
    def insert_signature(
        self,
        table_name: str,
        embedding: List[float],
        signature: Dict
    ) -> bool:
        """
        Insert a table signature into Milvus.
        
        Args:
            table_name: Name of the table
            embedding: Embedding vector
            signature: Signature dictionary
            
        Returns:
            True if insertion successful, False otherwise
        """
        try:
            logger.info(f"Inserting signature for table: {table_name}")
            
            if not self.collection:
                logger.error("Collection not initialized")
                return False
            
            # Prepare data
            signature_json = json.dumps(signature)
            created_at = datetime.now().isoformat()
            
            # Insert data
            entities = [
                [table_name],           # table_name
                [embedding],            # embedding
                [signature_json],       # signature_json
                [created_at]            # created_at
            ]
            
            self.collection.insert(entities)
            self.collection.flush()
            
            logger.info(f"Signature inserted successfully for: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting signature: {str(e)}")
            return False
    
    def search_similar(
        self,
        embedding: List[float],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar table signatures.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of top results to return (default from settings)
            threshold: Minimum similarity score (default from settings)
            
        Returns:
            List of dictionaries with keys: table_name, similarity_score, signature
        """
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            top_k = top_k or settings.similarity_top_k
            threshold = threshold or settings.similarity_threshold
            
            logger.info(f"Searching for similar signatures (top_k={top_k}, threshold={threshold})")
            
            # Define search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["table_name", "signature_json", "created_at"]
            )
            
            # Process results
            matches = []
            for hits in results:
                for hit in hits:
                    similarity_score = hit.score
                    
                    # Filter by threshold
                    if similarity_score >= threshold:
                        signature = json.loads(hit.entity.get('signature_json'))
                        
                        matches.append({
                            "table_name": hit.entity.get('table_name'),
                            "similarity_score": float(similarity_score),
                            "signature": signature,
                            "created_at": hit.entity.get('created_at')
                        })
            
            logger.info(f"Found {len(matches)} matches above threshold {threshold}")
            
            # Sort by similarity score (descending)
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching for similar signatures: {str(e)}")
            return []
    
    def delete_signature(self, table_name: str) -> bool:
        """
        Delete a signature by table name.
        
        Args:
            table_name: Name of the table to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            logger.info(f"Deleting signature for table: {table_name}")
            
            if not self.collection:
                logger.error("Collection not initialized")
                return False
            
            # Delete by expression
            expr = f'table_name == "{table_name}"'
            self.collection.delete(expr)
            self.collection.flush()
            
            logger.info(f"Signature deleted for: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting signature: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
            
            stats = self.collection.num_entities
            
            return {
                "collection_name": self.collection_name,
                "total_signatures": stats,
                "embedding_dimensions": self.embedding_dim
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}


# Global Milvus manager instance
milvus_manager = MilvusManager()
