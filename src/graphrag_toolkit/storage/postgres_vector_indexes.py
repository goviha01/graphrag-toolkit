# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import asyncio
from typing import List, Dict, Sequence, Any

from llama_index.core.schema import QueryBundle, BaseNode
from pgvector.psycopg import register_vector

import psycopg
import numpy as np

from graphrag_toolkit import EmbeddingType, GraphRAGConfig
from graphrag_toolkit.storage.vector_index import VectorIndex, to_embedded_query
from graphrag_toolkit.utils.embedding_utils import embed_nodes

logger = logging.getLogger(__name__)

def create_postgres_connection(connection_string):
    """Create PostgreSQL connection with pgvector extension enabled"""
    conn = psycopg.connect(connection_string)
    register_vector(conn)
    return conn

def create_table_if_not_exists(connection_string, table_name, dimensions):
    """Create a table with vector support if it doesn't exist"""
    conn = create_postgres_connection(connection_string)
    cur = conn.cursor()
    
    # Check if the pgvector extension is installed
    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    if not cur.fetchone():
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create table with vector column and appropriate indices
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding vector({dimensions})
        )
    """)
    
    # Create index using HNSW for faster approximate nearest neighbor search
    try:
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table_name}_vector_idx 
            ON {table_name} USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=64)
        """)
    except psycopg.errors.UndefinedObject:
        # If HNSW is not available, fall back to ivfflat
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table_name}_vector_idx 
            ON {table_name} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists=100)
        """)
    
    conn.commit()
    cur.close()
    conn.close()

class PostgresVectorIndex(VectorIndex):
    """PostgreSQL vector index implementation using pgvector"""

    @staticmethod
    def for_index(index_name, connection_string, embed_model=None, dimensions=None):
        """
        Factory method to create a PostgreSQL vector index
        
        Args:
            index_name: Name of the index (used as table name)
            connection_string: PostgreSQL connection string
            embed_model: Embedding model to use
            dimensions: Dimensions of embeddings
            
        Returns:
            PostgresVectorIndex instance
        """
        embed_model = embed_model or GraphRAGConfig.embed_model
        dimensions = dimensions or GraphRAGConfig.embed_dimensions
        
        create_table_if_not_exists(connection_string, index_name, dimensions)
        
        return PostgresVectorIndex(
            index_name=index_name, 
            connection_string=connection_string, 
            dimensions=dimensions, 
            embed_model=embed_model
        )
    
    class Config:
        arbitrary_types_allowed = True
    
    connection_string: str
    index_name: str
    dimensions: int
    embed_model: EmbeddingType
    
    def add_embeddings(self, nodes: Sequence[BaseNode]) -> Sequence[BaseNode]:
        """
        Add embeddings for the given nodes to the PostgreSQL vector store
        
        Args:
            nodes: Sequence of nodes to embed and store
            
        Returns:
            The input nodes
        """
        if not nodes:
            return nodes
            
        id_to_embed_map = embed_nodes(nodes, self.embed_model)
        
        conn = create_postgres_connection(self.connection_string)
        cur = conn.cursor()
        
        for node in nodes:
            node_id = node.node_id
            embedding = id_to_embed_map[node_id]
            
            # Convert content and metadata to appropriate format
            content = node.text
            metadata = node.metadata
            
            # Upsert node data with embedding
            cur.execute(
                f"""
                INSERT INTO {self.index_name} (id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (node_id, content, psycopg.adapt(metadata).getquoted().decode(), embedding)
            )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return nodes
    
    def top_k(self, query_bundle: QueryBundle, top_k: int = 5) -> Sequence[Dict[str, Any]]:
        """
        Find top-k similar nodes for the given query
        
        Args:
            query_bundle: Query to find similar nodes for
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with node data and similarity scores
        """
        # Ensure query has embedding
        query_bundle = to_embedded_query(query_bundle, self.embed_model)
        
        conn = create_postgres_connection(self.connection_string)
        cur = conn.cursor()
        
        # Query for top k nearest neighbors
        cur.execute(
            f"""
            SELECT id, content, metadata, 
                   1 - (embedding <=> %s) as similarity
            FROM {self.index_name}
            ORDER BY similarity DESC
            LIMIT %s
            """,
            (query_bundle.embedding.tolist(), top_k)
        )
        
        results = []
        for row in cur.fetchall():
            node_id, content, metadata_str, similarity = row
            
            # Convert string metadata back to dictionary
            try:
                import json
                metadata = json.loads(metadata_str)
            except:
                metadata = {}
                
            results.append({
                "node_id": node_id,
                "text": content,
                "metadata": metadata,
                "score": float(similarity)
            })
        
        cur.close()
        conn.close()
        
        return results
    
    def get_embeddings(self, ids: List[str] = None) -> Sequence[Dict[str, Any]]:
        """
        Retrieve embeddings for the specified node IDs
        
        Args:
            ids: List of node IDs to retrieve. If empty, returns all embeddings.
            
        Returns:
            Sequence of dictionaries with node data and embeddings
        """
        conn = create_postgres_connection(self.connection_string)
        cur = conn.cursor()
        
        if ids and len(ids) > 0:
            placeholders = ','.join(['%s'] * len(ids))
            cur.execute(
                f"""
                SELECT id, content, metadata, embedding
                FROM {self.index_name}
                WHERE id IN ({placeholders})
                """, 
                tuple(ids)
            )
        else:
            cur.execute(
                f"""
                SELECT id, content, metadata, embedding
                FROM {self.index_name}
                """
            )
        
        results = []
        for row in cur.fetchall():
            node_id, content, metadata_str, embedding = row
            
            # Convert string metadata back to dictionary
            try:
                import json
                metadata = json.loads(metadata_str)
            except:
                metadata = {}
            
            results.append({
                "node_id": node_id,
                "text": content,
                "metadata": metadata,
                "embedding": np.array(embedding)
            })
        
        cur.close()
        conn.close()
        
        return results