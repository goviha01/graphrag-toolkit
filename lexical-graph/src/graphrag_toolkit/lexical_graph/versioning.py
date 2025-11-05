# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, Union, List, Optional

VALID_FROM = '__aws__versioning__valid_from__'
VALID_TO = '__aws__versioning__valid_to__'
EXTRACT_TIMESTAMP = '__aws__versioning__extract_timestamp__'
BUILD_TIMESTAMP = '__aws__versioning__build_timestamp__'
VERSION_INDEPENDENT_ID_FIELDS = '__aws__versioning__id_fields__'

VERSIONING_METADATA_KEYS = [VALID_FROM, VALID_TO, EXTRACT_TIMESTAMP, BUILD_TIMESTAMP, VERSION_INDEPENDENT_ID_FIELDS]

TIMESTAMP_LOWER_BOUND = -1
TIMESTAMP_UPPER_BOUND = 10000000000000

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

class VersioningConfig():
    def __init__(self, at_timestamp:Optional[int]=None, enabled:Optional[bool]=None):
        self.at_timestamp = at_timestamp or TIMESTAMP_UPPER_BOUND
        if enabled is not None:
            self.enabled = enabled
        elif at_timestamp is not None:
            self.enabled = True
        else:
            self.enabled = False
