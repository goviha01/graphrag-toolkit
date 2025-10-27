# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from tqdm import tqdm
from typing import Any, List, Union, Dict

from graphrag_toolkit.lexical_graph.metadata import VALID_TO
from graphrag_toolkit.lexical_graph.indexing.node_handler import NodeHandler
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph_store_factory import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore, DummyVectorIndex, VectorIndex
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory
from graphrag_toolkit.lexical_graph.storage.constants import DEFAULT_EMBEDDING_INDEXES, INDEX_KEY, VIID_FIELD_KEY

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

GraphInfoType = Union[str, GraphStore]
VectorStoreInfoType = Union[str, VectorStore]

class VersionManager(NodeHandler):

    @staticmethod
    def for_graph_and_vector_store(graph_info:GraphInfoType=None, vector_store_info:VectorStoreInfoType=None, index_names=DEFAULT_EMBEDDING_INDEXES, **kwargs):

        graph_store = None
        vector_store = None
        
        if isinstance(graph_info, GraphStore):
            graph_store=graph_info
        else:
            graph_store=GraphStoreFactory.for_graph_store(graph_info, **kwargs)

        if isinstance(vector_store_info, VectorStore):
            vector_store=vector_store_info
        else:
            vector_store=VectorStoreFactory.for_vector_store(vector_store_info, index_names, **kwargs)

        return VersionManager(graph_store=graph_store, vector_store=vector_store)
    
    graph_store: GraphStore 
    vector_store: VectorStore

    def _get_source_ids(self, node:BaseNode) -> List[str]:

        viid_field_key = node.metadata.get(VIID_FIELD_KEY, None)

        if not viid_field_key:
            return []
        
        viid_field_value = node.metadata.get('source', {}).get('metadata', {}).get(viid_field_key, None)

        if not viid_field_value:
            return []
        
        cypher = f'''// find source node to be versioned
        MATCH (s:`__Source__`)
        WHERE s.{viid_field_key} = $viid_field_value
        AND s.{VALID_TO} = -1
        RETURN {self.graph_store.node_id("s.sourceId")} AS sourceId
        '''

        parameters = {
            'viid_field_value': viid_field_value
        }

        results = self.graph_store.execute_query(cypher, parameters)

        return [result['sourceId'] for result in results]
    
    def _get_node_ids(self, index:VectorIndex, source_ids:List[str]) -> Dict[str, List[str]]:
        
        if isinstance(index, DummyVectorIndex):
            return []
        
        if index.index_name == 'chunk':
            cypher = f'''// get chunk ids to be versioned
            MATCH (s)<-[:`__EXTRACTED_FROM__`]-(c)
            WHERE {self.graph_store.node_id("s.sourceId")} IN $sourceIds
            RETURN {{
                sourceId: {self.graph_store.node_id("s.sourceId")},
                nodeIds: collect({self.graph_store.node_id("c.chunkId")})
            }} AS result
            '''
        elif index.index_name == 'topic':
            cypher = f'''// get topic ids to be versioned
            MATCH (s)<-[:`__EXTRACTED_FROM__`]-()<-[:`__MENTIONED_IN__`]-(t)
            WHERE {self.graph_store.node_id("s.sourceId")} IN $sourceIds
            RETURN {{
                sourceId: {self.graph_store.node_id("s.sourceId")},
                nodeIds: collect({self.graph_store.node_id("t.topicId")})
            }} AS result
            '''
        elif index.index_name == 'statement':
            cypher = f'''// get topic ids to be versioned
            MATCH (s)<-[:`__EXTRACTED_FROM__`]-()<-[:`__MENTIONED_IN__`]-()<-[:`__BELONGS_TO__`]-(l)
            WHERE {self.graph_store.node_id("s.sourceId")} IN $sourceIds
            RETURN {{
                sourceId: {self.graph_store.node_id("s.sourceId")},
                nodeIds: collect({self.graph_store.node_id("l.statementId")})
            }} AS result
            '''
        else:
            raise ValueError(f'Invalid index name: {index.index_name}')
        
        parameters = {
            'sourceIds': source_ids
        }

        results = self.graph_store.execute_query(cypher, parameters)

        return {result['result']['sourceId']:result['result']['nodeIds'] for result in results}
    
    def _set_source_version_info(self, source_id:str, versioning_timestamp:int):

        cypher = f'''// set version info for source node
        MATCH (s:`__Source__`)
        WHERE {self.graph_store.node_id("s.sourceId")} = $sourceId
        SET s.{VALID_TO} = $versioningTimestamp
        '''

        properties = {
            'sourceId': source_id,
            'versioningTimestamp': versioning_timestamp
        }

        self.graph_store.execute_query_with_retry(cypher, properties)


    def accept(self, nodes: List[BaseNode], **kwargs: Any):
        
        versioning_timestamp = kwargs.get('versioning_timestamp', None)
        node_iterable = nodes if not self.show_progress else tqdm(nodes, desc=f'Applying version updates [versioning_timestamp: {versioning_timestamp}]')
        
        for node in node_iterable:

            if versioning_timestamp:

                if [key for key in [INDEX_KEY] if key in node.metadata]:
                    index_name = node.metadata[INDEX_KEY]['index']
                    if index_name == 'source':
                        source_ids = self._get_source_ids(node)
                        if source_ids:
                            for index in self.vector_store.all_indexes():
                                node_id_map = self._get_node_ids(index, source_ids)
                                for source_id, node_ids in node_id_map.items():
                                    logger.debug(f'Updating version info for {source_id} in vector index [index: {index.underlying_index_name()}, num_embeddings: {len(node_ids)}, versioning_timestamp: {versioning_timestamp}]')
                                    index.update_versioning(versioning_timestamp, node_ids)
                                    self._set_source_version_info(source_id, versioning_timestamp)
                        

            yield node

        

        