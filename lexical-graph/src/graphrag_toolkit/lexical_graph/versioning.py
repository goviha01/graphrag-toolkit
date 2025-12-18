# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Dict, Any, Union, List, Optional, Callable, Generator
from enum import Enum

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

logger = logging.getLogger(__name__)

VALID_FROM = '__aws__versioning__valid_from__'
VALID_TO = '__aws__versioning__valid_to__'
EXTRACT_TIMESTAMP = '__aws__versioning__extract_timestamp__'
BUILD_TIMESTAMP = '__aws__versioning__build_timestamp__'
VERSION_INDEPENDENT_ID_FIELDS = '__aws__versioning__id_fields__'
PREV_VERSIONS = '__aws__versioning__prev_versions__'

VERSIONING_METADATA_KEYS = [VALID_FROM, VALID_TO, EXTRACT_TIMESTAMP, BUILD_TIMESTAMP, VERSION_INDEPENDENT_ID_FIELDS, PREV_VERSIONS]

TIMESTAMP_LOWER_BOUND = -1
TIMESTAMP_UPPER_BOUND = 10000000000000

class VersioningMode(Enum): 
    NO_VERSIONING = 1
    CURRENT = 2
    PREVIOUS = 3
    AT_TIMESTAMP = 4
    BEFORE_TIMESTAMP = 5
    ON_OR_AFTER_TIMESTAMP = 6

IdFieldsType = Union[str, List[str]]

def add_versioning_info(
        metadata:Dict[str, Any],
        id_fields:Optional[IdFieldsType]=None,
        valid_from:Optional[int]=None
    ) -> Dict[str, Any]:
    if id_fields:
        metadata[VERSION_INDEPENDENT_ID_FIELDS] = id_fields if isinstance(id_fields, list) else [id_fields]
    if valid_from:
        metadata[VALID_FROM] = valid_from
    return metadata

def to_versioning_config(enable_versioning:bool):
    if enable_versioning:
        return VersioningConfig(versioning_mode=VersioningMode.CURRENT)
    else:
        return VersioningConfig(versioning_mode=VersioningMode.NO_VERSIONING)
    
class DeletePrevVersions():

    def __init__(self, lexical_graph:Any,  filter_fn:Callable[[Dict[str, Any]], bool]=None):
        self.lexical_graph = lexical_graph
        self.filter_fn = filter_fn or (lambda d: True)

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> Generator[BaseNode, None, None]:
        for node in nodes:
            j = json.loads(node.to_json())
            metadata = j['metadata']
            if 'aws::graph::index' in metadata:
                node_type = metadata['aws::graph::index']['index']
                source_metadata = metadata.get('source', {}).get('metadata', {})
                if node_type == 'source' and self.filter_fn(source_metadata):
                    prev_versions = (source_metadata.get('prev_versions', []))
                    if prev_versions:
                        logger.debug(f'Deleting previous versions for source with metadata {json.dumps(source_metadata)} [prev_versions: {prev_versions}]')
                        self.lexical_graph.delete_sources(source_ids=prev_versions)
                
            yield node

class VersioningConfig():
    def __init__(self, versioning_mode:Optional[VersioningMode]=None, at_timestamp:Optional[int]=None):

        if versioning_mode and at_timestamp:
            self.versioning_mode = versioning_mode
            self.at_timestamp = at_timestamp
        elif not versioning_mode and not at_timestamp:
            self.versioning_mode = VersioningMode.NO_VERSIONING
            self.at_timestamp = TIMESTAMP_UPPER_BOUND
        elif not versioning_mode:
            self.versioning_mode = VersioningMode.AT_TIMESTAMP
            self.at_timestamp = at_timestamp
        elif not at_timestamp:
            self.versioning_mode = versioning_mode
            self.at_timestamp = TIMESTAMP_UPPER_BOUND

    def apply(self, filter_config:FilterConfig) -> FilterConfig:

        if self.versioning_mode == VersioningMode.NO_VERSIONING:
            return filter_config
        
        if self.versioning_mode == VersioningMode.CURRENT:
            version_filter = MetadataFilter(
                key=VALID_TO,
                value=TIMESTAMP_UPPER_BOUND,
                operator=FilterOperator.EQ
            )
        elif self.versioning_mode == VersioningMode.PREVIOUS:
            version_filter = MetadataFilter(
                key=VALID_TO,
                value=TIMESTAMP_UPPER_BOUND,
                operator=FilterOperator.LT
            )
        elif self.versioning_mode == VersioningMode.AT_TIMESTAMP:
            version_filter = MetadataFilters(
                filters = [
                    MetadataFilter(
                        key=VALID_FROM,
                        value=self.at_timestamp,
                        operator=FilterOperator.LTE
                    ),
                    MetadataFilter(
                        key=VALID_TO,
                        value=self.at_timestamp,
                        operator=FilterOperator.GT
                    )
                ],
                condition = FilterCondition.AND
            )
        elif self.versioning_mode == VersioningMode.BEFORE_TIMESTAMP:
            version_filter = MetadataFilter(
                key=VALID_TO,
                value=self.at_timestamp,
                operator=FilterOperator.LT
            )
        elif self.versioning_mode == VersioningMode.ON_OR_AFTER_TIMESTAMP:
             version_filter = MetadataFilter(
                key=VALID_FROM,
                value=self.at_timestamp,
                operator=FilterOperator.GTE
            )
             
        if not filter_config.source_filters:
            return FilterConfig(version_filter)
        else:
            versionioned_filters = MetadataFilters(
                filters = [version_filter, filter_config.source_filters],
                condition = FilterCondition.AND
            )
            return FilterConfig(versionioned_filters)

        
        

