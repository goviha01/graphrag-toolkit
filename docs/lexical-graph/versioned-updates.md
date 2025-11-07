[[Home](./)]

## Versioned Updates

### Topics

  - [Overview](#overview)
  - [Document subgraphs](#document-subgraphs)
    - [Stable document identities](#stable-document-identities)
  - [Upgrading existing graph and vector stores](#upgrading-existing-graph-and-vector-stores)
    - [Upgrading specific tenants](#upgrading-specific-tenants)
    - [Upgrading specific vector indexes](#upgrading-specific-vector-indexes)

### Overview

The graphrag-toolkit allows you to version source documents along a single timeline. Using this versioning feature, if you re-ingest a document whose contents and/or metadata have changed, any old documents will be archived, and the newly ingested document treated as the current version of the source document.

### Document subgraphs

The `(source)<--(chunk)<--(topic)<--(statement)` part of the lexical graph model represents a bounded document subgraph. The id of a source node is a function of the metadata and textual contents of a source document. The ids of chunks, topics, and statements are in turn a function of the source id. If the metadata and/or contents of a source document change, and the document is reprocessed, the source will be assigned a different id – and so will all the chunks, topics and statements that derive from that source.

This means that if you extract two different versions of a document (i.e. versions with different contents and/or metadata), you’ll end up with two different bounded document subgraphs: two source nodes, and then independent `(chunk)<--(topic)<--(statement)` subgraphs beneath each of those source nodes. If the toolkit's versioning feature is enabled, the last version of the document to be ingested will be treated as the current version, and all other versions marked as historical, archived versions.

#### Stable document identities

For a document to be versioned in this manner, there must be some way of specifying that different sets of text and metadata represent _different_ versions of the _same_ document. In other words, the document must have a stable identity, independent of variations in content and/or metadata. 

The graphrag-toolkit uses a concept of _version-independent metadata field values_ to represent this stable identify. When you index a document, you can specify which of that document's metadata fields represent its stable identify. For example, if a document has `title`, `author` and `last_updated` metadata fields, you might specify that a combination of the `title` and `author` metadata fields represent that document's stable identify. When the document is indexed, any previously indexed, non-versioned documents whose `title` and `author` field _values_ match those of the newly ingested document will be archived.

### Upgrading existing graph and vector stores

If you have existing graph and vector stores created by a version of the graphrag-toolkit prior to version 3.14, you will need to upgrade them before using the versioned updates feature. The graphrag-toolkit includes an `upgrade_for_versioning.py` script that will upgrade a graph and vector store so that you can use versioned updates.

> Do not index any documenst while the upgrade script is running.

Download the [`upgrade_for_versioning.py`](https://github.com/awslabs/graphrag-toolkit/blob/main/examples/lexical-graph/scripts/upgrade_for_versioning.py) script to an environment that can access your graph and vector stores. Then run:

```
python upgrade_for_versioning.py --graph-store <graph_store_info> --vector_store <vector_store_info>
```

#### Upgrading specific tenants

By default, the script upgrades all [tenants](./multi-tenancy.md) in the graph and vector stores. You can restrict the list of tenants using the `--tenant-ids <space_separated_tenant_ids>` parameter. For example:

```
python upgrade_for_versioning.py --graph-store <graph_store_info> --vector_store <vector_store_info> --tenant-ids t1 t2 _default
```

Note that `_default` identifies the default tenant.

#### Upgrading specific vector indexes

By default, the script only updates the chunk index for each tenant. Your vector store may also contain a statement index, which is used by the [semantic-guide search](./semantic-guided-search.md). Semantic-guided search is likely to be removed in future versions of the toolkit – to avoid unnecessary work, we therefore recommend _not_ upgrading this index.

If, however, you do want to upgrade the statement index, supply an `--index-names <space_separated_index_names>` parameter:

```
python upgrade_for_versioning.py --graph-store <graph_store_info> --vector_store <vector_store_info> --index_names chunk statement
```


Configuring versioned updates
Indexing
Querying