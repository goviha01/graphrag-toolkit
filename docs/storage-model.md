[[Home](./)]

## Storage Model

### Topics

- [Overview](#overview)
- [Graph store](#graph-store)
  - [Neptune Database and Neptune Analytics graph stores](#neptune-database-and-neptune-analytics-graph-stores)
- [Vector store](#vector-store)
  - [Amazon OpenSearch Serverless and Neptune Analytics vector stores](#amazon-opensearch-serverless-and-neptune-analytics-vector-stores)

### Overview

The graphrag-toolkit uses two separate stores: a `GraphStore` and a `VectorStore`. A `VectorStore` acts as a container for a collection of `VectorIndex`. When constructing or querying a graph, you must provide instances of both a graph store and vector store.

The toolkit provides graph store implementations for both [Amazon Neptune Analytics](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html) and [Amazon Neptune Database](https://docs.aws.amazon.com/neptune/latest/userguide/intro.html), and now [FalkorDB](https://docs.falkordb.com/)**,** along with vector store implementations for Neptune Analytics and [Amazon OpenSearch Serverless](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html). The graphrag-toolkit provides several convenient factory methods for creating instances of these stores. These factory methods accept formatted store identifiers, described below.

> This early release of the toolkit provides support for Amazon Neptune and Amazon OpenSearch Serverless, but we welcome alternative store implementations. The store APIs and the ways in which the stores are used have been designed to anticipate alternative implementations. However, the proof is in the development: if you experience issues developing an alternative store, [let us know](https://github.com/awslabs/graphrag-toolkit/issues).

Graph stores and vector stores provide connectivity to an *existing* storage instance, which you will need to have provisioned beforehand.

#### Code examples

The code examples here are formatted to run in a Jupyter notebook. If you’re building an application with a main entry point, put your application logic inside a method, and add an [`if __name__ == '__main__'` block](./faq.md#runtimeerror-please-use-nest_asyncioapply-to-allow-nested-event-loops).

### Graph store

Graph stores must support the [openCypher](https://opencypher.org/) property graph query language. Graph construction queries typically use an `UNWIND ... MERGE` idiom to create or update the graph for a [batch of inputs](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/best-practices-content.html#best-practices-content-14). The Neptune graph store implementations override the `GraphStore.node_id()` method to ensure that node ids in the code (e.g. `chunkId`) are mapped to Neptune's `~id` reserved property. Alternative graph store implementations can leave the base implementation of `node_id()` as-is. This will result in node ids being mapped to a property of the same name (i.e. a reference to `chunkId` in the code will be mapped to a `chunkId` property of a node).

#### Neptune Database and Neptune Analytics graph stores

You can use the `GraphStoreFactory.for_graph_store()` static factory method to create an instance of a Neptune Analytics or Neptune Database graph store.

To create a Neptune Database graph store, supply a connection string that begins `neptune-db://`, followed by an [endpoint](https://docs.aws.amazon.com/neptune/latest/userguide/feature-overview-endpoints.html):

```python
from graphrag_toolkit.storage import GraphStoreFactory

neptune_connection_info = 'neptune-db://mydbcluster.cluster-123456789012.us-east-1.neptune.amazonaws.com:8182'

graph_store = GraphStoreFactory.for_graph_store(neptune_connection_info)
```

To create a Neptune Analytics graph store, supply a connection string that begins `neptune-graph://`, followed by the graph's identifier:

```
from graphrag_toolkit.storage import GraphStoreFactory

neptune_connection_info = 'neptune-graph://g-jbzzaqb209'

graph_store = GraphStoreFactory.for_graph_store(neptune_connection_info)
```

#### FalkorDB graph store

You can now use the `GraphStoreFactory.for_graph_store()` static factory method to create an instance of a FalkorDB graph store.

The FalkorDB graph store currently supports the [SemanticGuidedRetriever](./querying.md#semanticguidedretriever). It does not support the [TraversalBasedRetriever](./querying.md#traversalbasedretriever).

To create a [FalkorDB Cloud](https://app.falkordb.cloud/) graph store, supply a connection string that begins `falkordb://`, followed by the FalkorDB endpoint:

```python
from graphrag_toolkit.storage import GraphStoreFactory

falkordb_connection_info = 'falkordb://your-falkordb-endpoint'

graph_store = GraphStoreFactory.for_graph_store(falkordb_connection_info)
```

You may also need to pass a username and password, and specify whether or not to use SSL:

```python
from graphrag_toolkit.storage import GraphStoreFactory

falkordb_connection_info = 'falkordb://<your-falkordb-endpoint>'

graph_store = GraphStoreFactory.for_graph_store(
    falkordb_connection_info,
    username='<username>',
    password='<password>',
    ssl=True
)
```

To create a local FalkorDB graph store, supply a connection string that has only `falkordb://`;

```python
from graphrag_toolkit.storage import GraphStoreFactory

falkordb_connection_info = 'falkordb://'

graph_store = GraphStoreFactory.for_graph_store(falkordb_connection_info)
```

### Vector store

A vector store is a collection of vector indexes. The graphrag-toolkit uses up to two vector indexes: a chunk index and a statement index. The chunk index is typically much smaller than the statement index. If you want to use the [SemanticGuidedRetriever](./querying.md#semanticguidedretriever), you will need to enable the statement index. If you want to use the [TraversalBasedRetriever](./querying.md#traversalbasedretriever), you will need to enable the chunk index. If you want to use both retrievers, you will need to enable both indexes. (The `VectorStoreFactory` described below enables both indexes by default.)

#### Amazon OpenSearch Serverless and Neptune Analytics vector stores

You can use the `VectorStoreFactory.for_vector_store()` static factory method to create an instance of an Amazon OpenSearch Serverless or Neptune Database vector store.

To create an Amazon OpenSearch Serverless vector store, supply a connection string that begins `aoss://`, followed the https endpoint of the OpenSearch Serverless collection:

```python
from graphrag_toolkit.storage import VectorStoreFactory

opensearch_connection_info = 'aoss://https://123456789012.us-east-1.aoss.amazonaws.com'

vector_store = VectorStoreFactory.for_vector_store(opensearch_connection_info)
```

To create a Neptune Analytics vector store, supply a connection string that begins `neptune-graph://`, followed by the graph's identifier:

```python
from graphrag_toolkit.storage import VectorStoreFactory

neptune_connection_info = 'neptune-graph://g-jbzzaqb209'

vector_store = VectorStoreFactory.for_vector_store(neptune_connection_info)
```

#### PostgreSQL vector store

You can use the `VectorStoreFactory.for_vector_store()` or `VectorStoreFactory.for_postgres()` static factory method to create an instance of a PostgreSQL vector store.

To create a PostgreSQL vector store, supply a connection string that begins with `postgres://`, followed by the PostgreSQL connection details:

```python
from graphrag_toolkit.storage import VectorStoreFactory
from graphrag_toolkit.storage.constants import POSTGRES_VECTOR

# Using for_vector_store method
postgres_connection_info = f"{POSTGRES_VECTOR}postgresql://username:password@hostname:port/database"
vector_store = VectorStoreFactory.for_vector_store(postgres_connection_info)

# Or using the dedicated method
vector_store = VectorStoreFactory.for_postgres(
    connection_string="postgresql://username:password@hostname:port/database",
    index_names=["chunk", "statement"]
)
```

By default, the `VectorStoreFactory` will enable both the statement index and the chunk index. If you want to enable just one of the indexes, pass an `index_names` argument to the factory method:

```
vector_store = VectorStoreFactory.for_vector_store(opensearch_connection_info, index_names=['chunk'])
```
