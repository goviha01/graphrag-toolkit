# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from tqdm import tqdm
from typing import Any, List, Union

from graphrag_toolkit.lexical_graph.indexing.node_handler import NodeHandler
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph_store_factory import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory
from graphrag_toolkit.lexical_graph.storage.constants import DEFAULT_EMBEDDING_INDEXES

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

    def accept(self, nodes: List[BaseNode], **kwargs: Any):
        
        node_iterable = nodes if not self.show_progress else tqdm(nodes, desc='Applying version updates')
        
        for node in node_iterable:


            yield node

        

        