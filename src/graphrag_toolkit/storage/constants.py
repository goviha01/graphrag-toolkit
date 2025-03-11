# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

INDEX_KEY = 'aws::graph::index'
ALL_EMBEDDING_INDEXES = ['chunk', 'statement', 'topic', 'fact']
DEFAULT_EMBEDDING_INDEXES = ['chunk', 'statement']

# Vector store types
OPENSEARCH_SERVERLESS = 'opensearch'
NEPTUNE_ANALYTICS = 'neptune'
POSTGRES_VECTOR = 'postgres'
DUMMY_VECTOR_STORE = 'dummy'
