# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.indexing.model import Fact
from graphrag_toolkit.storage.graph_store import GraphStore
from graphrag_toolkit.storage.graph_utils import search_string_from, label_from, relationship_name_from
from graphrag_toolkit.indexing.build.graph_builder import GraphBuilder
from graphrag_toolkit.indexing.constants import DEFAULT_CLASSIFICATION

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class FactGraphBuilder(GraphBuilder):
    
    @classmethod
    def index_key(cls) -> str:
        return 'fact'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
            
        fact_metadata = node.metadata.get('fact', {})
        include_domain_labels = kwargs.pop('include_domain_labels', True)

        if fact_metadata:

            fact = Fact.model_validate(fact_metadata)
        
            logger.debug(f'Inserting fact [fact_id: {fact.factId}]')

            statements = [
                '// insert facts',
                'UNWIND $params AS params'
            ]

            
            statements.extend([
                f'MATCH (statement:`__Statement__`{{{graph_client.node_id("statementId")}: params.statement_id}})',
                f'MERGE (fact:`__Fact__`{{{graph_client.node_id("factId")}: params.fact_id}})',
                'ON CREATE SET fact.relation = params.p, fact.value = params.fact ON MATCH SET fact.relation = params.p, fact.value = params.fact',
                'MERGE (fact)-[:`__SUPPORTS__`]->(statement)',
            ])

            statements.extend([
                'WITH fact, params',
                f'MATCH (subject:`__Entity__`{{{graph_client.node_id("entityId")}: params.s_id}})'
                'MERGE (subject)-[:`__SUBJECT__`]->(fact)'
            ])

            properties = {
                'statement_id': fact.statementId,
                'fact_id': fact.factId,
                's_id': fact.subject.entityId,
                'fact': node.text
            }

            if fact.object:

                statements.extend([
                    'WITH fact, params',
                    f'MATCH (object:`__Entity__`{{{graph_client.node_id("entityId")}: params.o_id}})'
                    'MERGE (object)-[:`__OBJECT__`]->(fact)'
                ])

                properties.update({                
                    'o_id': fact.object.entityId
                })
        
            query = '\n'.join(statements)
                
            graph_client.execute_query_with_retry(query, self._to_params(properties), max_attempts=5, max_wait=7)

            statements = [
                '// insert connection to prev facts',
                'UNWIND $params AS params'
            ]

            statements.extend([
                f'MATCH (fact:`__Fact__`{{{graph_client.node_id("factId")}: params.fact_id}})<-[:`__SUBJECT__`]-(:`__Entity__`)-[:`__OBJECT__`]->(prevFact:`__Fact__`)',
                'MERGE (fact)<-[:`__NEXT__`]-(prevFact)'
            ])

            properties = {
                'fact_id': fact.factId
            }

            query = '\n'.join(statements)
                
            graph_client.execute_query_with_retry(query, self._to_params(properties))

            if fact.object:

                statements = [
                    '// insert connection to next facts',
                    'UNWIND $params AS params'
                ]
            
                statements.extend([
                    f'MATCH (fact:`__Fact__`{{{graph_client.node_id("factId")}: params.fact_id}})<-[:`__OBJECT__`]-(:`__Entity__`)-[:`__SUBJECT__`]->(nextFact:`__Fact__`)',
                    'MERGE (fact)-[:`__NEXT__`]->(nextFact)'
                ])

                properties = {
                    'fact_id': fact.factId
                }

                query = '\n'.join(statements)
                    
                graph_client.execute_query_with_retry(query, self._to_params(properties))
           

        else:
            logger.warning(f'fact_id missing from fact node [node_id: {node.node_id}]')