# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import argparse
import json
import logging
import sys
import json
import time

from itertools import islice
from tqdm import tqdm
from typing import List, Dict

from graphrag_toolkit.lexical_graph import to_tenant_id, DEFAULT_TENANT_NAME
from graphrag_toolkit.lexical_graph.versioning import VALID_FROM, VALID_TO, TIMESTAMP_LOWER_BOUND, TIMESTAMP_UPPER_BOUND
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage.graph import MultiTenantGraphStore, GraphStore
from graphrag_toolkit.lexical_graph.storage.graph import NonRedactedGraphQueryLogFormatting
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory
from graphrag_toolkit.lexical_graph.storage.vector import MultiTenantVectorStore, VectorStore, VectorIndex, DummyVectorIndex

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def iter_batch(iterable, batch_size):
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, batch_size))
        if len(b) == 0:
            break
        yield b

def get_num_source_ids(graph_store:GraphStore):
    
    cypher = f'''
    MATCH (source:`__Source__`) 
    RETURN count(source) AS source_count'''
    
    results = graph_store.execute_query_with_retry(cypher, {})
    
    return [r['source_count'] for r in results][0]

def get_source_ids(graph_store:GraphStore, failed_source_ids:List[str]):
    
    cypher = f'''
    MATCH (source:`__Source__`) 
    WHERE coalesce(source.{VALID_FROM}, -1) = -1
    AND coalesce(source.{VALID_TO}, -1) = -1
    AND NOT {graph_store.node_id("source.sourceId")} IN $failedSourceIds
    RETURN {graph_store.node_id("source.sourceId")} AS source_id LIMIT 10000'''

    parameters = {
        'failedSourceIds': failed_source_ids
    }
    
    results = graph_store.execute_query_with_retry(cypher, parameters)
    
    return [r['source_id'] for r in results]

def set_source_versioning_info(graph_store:GraphStore, source_ids):

    cypher = f'''
    MATCH (source:`__Source__`) 
    WHERE {graph_store.node_id("source.sourceId")} IN $sourceIds
    SET source.{VALID_FROM} = {TIMESTAMP_LOWER_BOUND}, source.{VALID_TO} = {TIMESTAMP_UPPER_BOUND}'''

    parameters = {
        'sourceIds': source_ids
    }
    
    graph_store.execute_query_with_retry(cypher, parameters, max_attempts=10, max_wait=7)

def get_node_ids(graph_store:GraphStore, index:VectorIndex, source_ids:List[str]) -> Dict[str, List[str]]:
        
        if isinstance(index, DummyVectorIndex):
            return []
        
        if index.index_name == 'chunk':
            cypher = f'''// get chunk ids to be versioned
            MATCH (s)<-[:`__EXTRACTED_FROM__`]-(c)
            WHERE {graph_store.node_id("s.sourceId")} IN $sourceIds
            WITH {graph_store.node_id("s.sourceId")} as source_id, collect({graph_store.node_id("c.chunkId")}) as node_ids
            RETURN {{
                sourceId: source_id,
                nodeIds: node_ids
            }} AS result
            '''
        elif index.index_name == 'topic':
            cypher = f'''// get topic ids to be versioned
            MATCH (s)<-[:`__EXTRACTED_FROM__`]-()<-[:`__MENTIONED_IN__`]-(t)
            WHERE {graph_store.node_id("s.sourceId")} IN $sourceIds
            WITH {graph_store.node_id("s.sourceId")} as source_id, collect({graph_store.node_id("t.topicId")}) as node_ids
            RETURN {{
                sourceId: source_id,
                nodeIds: node_ids
            }} AS result
            '''
        elif index.index_name == 'statement':
            cypher = f'''// get topic ids to be versioned
            MATCH (s)<-[:`__EXTRACTED_FROM__`]-()<-[:`__MENTIONED_IN__`]-()<-[:`__BELONGS_TO__`]-(l)
            WHERE {graph_store.node_id("s.sourceId")} IN $sourceIds
            WITH {graph_store.node_id("s.sourceId")} as source_id, collect({graph_store.node_id("l.statementId")}) as node_ids
            RETURN {{
                sourceId: source_id,
                nodeIds: node_ids
            }} AS result
            '''
        else:
            raise ValueError(f'Invalid index name: {index.index_name}')
        
        parameters = {
            'sourceIds': source_ids
        }

        results = graph_store.execute_query(cypher, parameters)

        return {result['result']['sourceId']:result['result']['nodeIds'] for result in results}


class VectorStoreUnitOfWork():

    def __init__(self, index:VectorIndex, stats, batch_size):
        self.index = index
        self.stats = stats
        self.batch_size = batch_size
        self.node_ids = []
        self.source_id_map = {}

    def add_node_ids(self, source_id, node_ids):
        self.node_ids.extend(node_ids)
        self.source_id_map.update({node_id:source_id for node_id in node_ids})

    @property
    def size(self):
        return len(self.node_ids)

    def apply(self):

        failed_source_ids = set()

        for node_id_batch in iter_batch(self.node_ids, batch_size=self.batch_size):
            
            failed_node_ids = self.index.enable_for_versioning(node_id_batch)
            failed_source_ids.update(self.source_id_map[failed_node_id] for failed_node_id in failed_node_ids)
            
            if self.index.index_name not in self.stats:
                self.stats[self.index.index_name] = {'succeeded': 0, 'failed': 0}
            
            self.stats[self.index.index_name]['succeeded'] += len(node_id_batch) - len(failed_node_ids)
            self.stats[self.index.index_name]['failed'] += len(failed_node_ids)

        self.node_ids = []
        self.source_id_map = {}

        return list(failed_source_ids)

class UnitOfWork():

    def __init__(self, vector_store:VectorStore, graph_store:GraphStore, stats, batch_size, progress):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.vector_store_units_of_work = {
            index.index_name:VectorStoreUnitOfWork(index, stats, batch_size)
            for index in vector_store.all_indexes()
        }
        self.batch_size = batch_size
        self.progress = progress
        self.source_ids = []
        self.failed_source_ids = []

    @property
    def size(self):
        return len(self.source_ids)

    def add_source_id(self, source_id):

        self.source_ids.append(source_id)

        for index in self.vector_store.all_indexes():          
            node_id_map = get_node_ids(self.graph_store, index, [source_id])
            for _, node_ids in node_id_map.items():
                self.vector_store_units_of_work[index.index_name].add_node_ids(source_id, node_ids)

        do_apply = False

        for vector_store_unit_of_work in self.vector_store_units_of_work.values():
            if vector_store_unit_of_work.size >= (self.batch_size * 10):
                do_apply = True

        if do_apply:
            return self.apply()
        else:
            return []

    def apply(self):
        failed_source_ids = set()

        for vector_store_unit_of_work in self.vector_store_units_of_work.values():
            failed_source_ids.update(vector_store_unit_of_work.apply())
        
        set_source_versioning_info(self.graph_store, [source_id for source_id in self.source_ids if source_id not in failed_source_ids])
       
        self.progress.update(len(self.source_ids))
        self.source_ids = []

        return list(failed_source_ids)


def upgrade(graph_store_info:str, vector_store_info:str, index_names:List[str], batch_size:int, tenant_id=None):

    with (
        GraphStoreFactory.for_graph_store(graph_store_info, log_formatting=NonRedactedGraphQueryLogFormatting()) as graph_store,
        VectorStoreFactory.for_vector_store(vector_store_info, index_names=index_names) as vector_store
    ):
        tenant_id = to_tenant_id(tenant_id)

        if tenant_id.is_default_tenant():
            print(f'Upgrading default tenant')
        else:
            print(f'Upgrading {tenant_id}')
            graph_store = MultiTenantGraphStore.wrap(
                graph_store,
                tenant_id
            )
            vector_store = MultiTenantVectorStore.wrap(
                vector_store,
                tenant_id
            )

        stats = {
            'tenant_id': str(tenant_id)
        }

        num_source_ids = get_num_source_ids(graph_store)

        progress_bar = tqdm(total=num_source_ids, desc=f'Upgrading sources for tenant {tenant_id}')

        failed_source_ids = []
        source_ids = get_source_ids(graph_store, failed_source_ids)

        while source_ids:

            unit_of_work = UnitOfWork(vector_store, graph_store, stats, batch_size, progress_bar)

            source_id = source_ids.pop() if source_ids else None
            while source_id:
                failed_source_ids.extend(unit_of_work.add_source_id(source_id))
                source_id = source_ids.pop() if source_ids else None

            failed_source_ids.extend(unit_of_work.apply())

            source_ids = get_source_ids(graph_store, failed_source_ids)

        stats['failed_source_ids'] = failed_source_ids

        return stats
    
def get_tenant_ids(graph_store_info):

    print('Getting tenant ids...')

    with (
        GraphStoreFactory.for_graph_store(graph_store_info, log_formatting=NonRedactedGraphQueryLogFormatting()) as graph_store,
    ):
    
        cypher = '''MATCH (n)<-[:`__EXTRACTED_FROM__`]-()
        WITH DISTINCT labels(n) as lbls
        WITH split(lbls[0], '__') AS lbl_parts WHERE size(lbl_parts) > 2
        WITH lbl_parts WHERE lbl_parts[1] = 'Source' AND lbl_parts[2] <> ''
        RETURN DISTINCT lbl_parts[2] AS tenant_id
        '''

        results = graph_store.execute_query(cypher)

        return [result['tenant_id'] for result in results]

def do_upgrade():

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-store', help = 'Graph store connection string')
    parser.add_argument('--vector-store', help = 'Vector store connection string')  
    parser.add_argument('--index-names', nargs='*', help = 'Space-separated list of index names (optional, default "chunk")')
    parser.add_argument('--tenant-ids', nargs='*', help = 'Space-separated list of tenant ids (optional)')
    parser.add_argument('--batch-size', nargs='?', help = 'Batch size (optional, default 100)')
    
    args, _ = parser.parse_known_args()
    args_dict = { k:v for k,v in vars(args).items() if v}

    graph_store_info = args_dict['graph_store']
    vector_store_info = args_dict['vector_store']
    index_names = args_dict.get('index_names', ['chunk'])
    tenant_ids = args_dict.get('tenant_ids', [])
    batch_size = int(args_dict.get('batch_size', 100))

    if not tenant_ids:
        tenant_ids = [DEFAULT_TENANT_NAME]
        tenant_ids.extend(get_tenant_ids(graph_store_info))
    
    print(f'graph_store_info               : {graph_store_info}')
    print(f'vector_store_info              : {vector_store_info}')
    print(f'index_names                    : {index_names}')
    print(f'tenant_ids                     : {tenant_ids}')
    print(f'batch_size                     : {batch_size}')
    
    print()

    results = []
    
    progress_bar_1 = tqdm(total=len(tenant_ids), desc='Upgrading sources')
    for tenant_id in tenant_ids:
        results.append(upgrade(graph_store_info, vector_store_info, index_names, batch_size, tenant_id))
        progress_bar_1.update(1)
                
    print()
    print(json.dumps(results, indent=2))
    
            
if __name__ == '__main__':
    start = time.time()
    do_upgrade()
    end = time.time()
    print(end - start)