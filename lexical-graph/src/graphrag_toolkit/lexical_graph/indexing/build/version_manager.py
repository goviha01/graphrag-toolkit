# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from tqdm import tqdm
from typing import Any, List, Union, Dict

from graphrag_toolkit.lexical_graph.metadata import VALID_FROM, VALID_TO, VERSIONING_FIELDS, format_versioning_fields
from graphrag_toolkit.lexical_graph.indexing.node_handler import NodeHandler
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph_store_factory import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore, DummyVectorIndex, VectorIndex
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory
from graphrag_toolkit.lexical_graph.storage.constants import DEFAULT_EMBEDDING_INDEXES, INDEX_KEY

from llama_index.core.schema import BaseNode
from llama_index.core.utils import iter_batch
from llama_index.core.schema import NodeRelationship

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

    def _get_other_source_versions(self, versioning_fields:List[str], node:BaseNode) -> List[Dict[str, Any]]:
        
        if not versioning_fields:
            return []
        
        formatted_versioning_fields = format_versioning_fields(versioning_fields)
        
        where_clauses = []
        parameters = {}

        for i, versioning_field in enumerate(versioning_fields):
            versioning_field_value = node.metadata.get('source', {}).get('metadata', {}).get(versioning_field, None)
            if versioning_field_value:
                param_name = f'param{i}'
                where_clauses.append(f's.{versioning_field} = ${param_name}')
                property_value_fn = self.graph_store.property_assigment_fn(versioning_field, versioning_field_value)
                parameters[param_name] = property_value_fn(versioning_field_value)

        if not where_clauses:
            return []
        
        where_clauses.append(f"coalesce(s.{VERSIONING_FIELDS}, '{formatted_versioning_fields}') = $versioningFields")
        parameters['versioningFields'] = formatted_versioning_fields
        
        where_clause = ' AND '.join(where_clauses)
        
        cypher = f'''// find source node to be versioned
        MATCH (s:`__Source__`)
        WHERE {where_clause}
        RETURN {{
            source_id: {self.graph_store.node_id("s.sourceId")},
            valid_from: coalesce(s.{VALID_FROM}, -1),
            valid_to: coalesce(s.{VALID_TO}, -1)
        }} AS result
        '''

        results = self.graph_store.execute_query(cypher, parameters)

        return [result['result'] for result in results]
    
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
    
    def _set_source_node_version_info(self, source_id:str, versioning_timestamp:int, versioning_fields:List[str]):

        cypher = f'''// set version info for old source node
        MATCH (s:`__Source__`)
        WHERE {self.graph_store.node_id("s.sourceId")} = $sourceId
        SET s.{VALID_TO} = $versioningTimestamp, s.{VERSIONING_FIELDS} = $versioningFields
        '''

        properties = {
            'sourceId': source_id,
            'versioningTimestamp': versioning_timestamp,
            'versioningFields': format_versioning_fields(versioning_fields)
        }

        self.graph_store.execute_query_with_retry(cypher, properties)

    def _update_vector_store_versions(self, source_id:str, node_ids:List[str], versioning_timestamp:int, index:VectorIndex):

        node_id_batches = [
            node_ids[x:x+100] 
            for x in range(0, len(node_ids), 100)
        ]

        logger.debug(f'Updating valid_to version info for {source_id} in vector index [index: {index.underlying_index_name()}, batch_sizes: {[len(b) for b in node_id_batches]}, valid_to: {versioning_timestamp}]')

        for node_id_batch in node_id_batches:
            index.update_versioning(versioning_timestamp, node_id_batch)


    def _is_latest(self, extract_timestamp:int, other_source_versions:List[Dict[str, Any]]) -> bool:

        for other_source_version in other_source_versions:
            if other_source_version['valid_from'] > extract_timestamp:
                return False
            
        return True
    
    def _get_lowest_valid_from(self, extract_timestamp:int, other_source_versions:List[Dict[str, Any]]) -> int:

        lowest_valid_from = None

        for other_source_version in other_source_versions:
            if other_source_version['valid_from'] > extract_timestamp:
                if not lowest_valid_from or other_source_version['valid_from'] < lowest_valid_from:
                    lowest_valid_from = other_source_version['valid_from']
        
        return lowest_valid_from or -1
    
    def _get_adjusted_previous_versions(self, extract_timestamp:int, other_source_versions:List[Dict[str, Any]]) -> Dict[str, int]:

        updateable_older_version_id = None
        updateable_older_version_valid_from = None

        for other_source_version in other_source_versions:
            if other_source_version['valid_from'] < extract_timestamp and other_source_version['valid_to'] > -1:
                if not updateable_older_version_valid_from or other_source_version['valid_from'] > updateable_older_version_valid_from:
                    updateable_older_version_id = other_source_version['source_id']
                    updateable_older_version_valid_from = other_source_version['valid_from']

        if updateable_older_version_id:
            return { updateable_older_version_id: extract_timestamp }
        else:
            return {}
    
    def accept(self, nodes: List[BaseNode], **kwargs: Any):
        
        node_iterable = nodes if not self.show_progress else tqdm(nodes, desc='Applying version updates')
        
        source_version_info = {}

        for node in node_iterable:

            if [key for key in [INDEX_KEY] if key in node.metadata]:

                index_name = node.metadata[INDEX_KEY]['index']

                if index_name == 'source':

                    previous_versions:Dict[str, int] = {}

                    source_id = node.metadata['source']['sourceId']
                    extract_timestamp = node.metadata['source']['versioning']['extract_timestamp']
                    versioning_fields = node.metadata.get('source', {}).get('versioning', {}).get('versioning_fields', None)

                    other_source_versions = self._get_other_source_versions(versioning_fields, node)

                    if self._is_latest(extract_timestamp, other_source_versions):
                        previous_versions.update({
                            other_source_version['source_id']:extract_timestamp
                            for other_source_version in other_source_versions 
                            if other_source_version['valid_to'] == -1
                        })
                    else:
                        lowest_valid_from = self._get_lowest_valid_from(extract_timestamp, other_source_versions)
                        logger.debug(f'Source is not the latest version, setting valid_to version info [source_id: {source_id}, valid_to: {lowest_valid_from}]')
                        node.metadata['source']['versioning']['valid_to'] = lowest_valid_from

                    previous_versions.update(self._get_adjusted_previous_versions(extract_timestamp, other_source_versions))

                    for source_id, valid_to in previous_versions.items():
                        for index in self.vector_store.all_indexes():
                            node_id_map = self._get_node_ids(index, [source_id])
                            for source_id, node_ids in node_id_map.items():
                                self._update_vector_store_versions(source_id, node_ids, valid_to, index)
                                self._set_source_node_version_info(source_id, valid_to, versioning_fields)
                    
                    node.metadata['source']['previous_versions'] = list(previous_versions.keys())

                    source_version_info[source_id] = {
                        'valid_from': node.metadata['source']['versioning']['valid_from'],
                        'valid_to': node.metadata['source']['versioning']['valid_to']
                    }

                    logger.debug(f'Source version info [source_id: {source_id}, version_info: {source_version_info[source_id]}]')

                else:

                    source_id = node.metadata.get('source', {}).get('sourceId', None)

                    if source_id:
                        version_info = source_version_info.get(source_id, None)

                        if version_info:
                            node.metadata['source']['versioning']['valid_from'] = version_info['valid_from']
                            node.metadata['source']['versioning']['valid_to'] = version_info['valid_to']
                        else:
                            logger.debug(f'No version info found for source {source_id}')
                        
            yield node

        

        