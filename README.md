## GraphRAG Toolkit

The graphrag-toolkit is a Python toolkit for building GraphRAG applications. It provides a framework for automating the construction of a graph from unstructured data, and composing question-answering strategies that query this graph when answering user questions. 

The toolkit uses low-level [LlamaIndex](https://docs.llamaindex.ai/en/stable/)  components – data connectors, metadata extractors, and transforms – to implement much of the graph construction process. By default, the toolkit uses [Amazon Neptune Analytics](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html) or [Amazon Neptune Database](https://docs.aws.amazon.com/neptune/latest/userguide/intro.html) for its graph store, and Neptune Analytics or [Amazon OpenSearch Serverless](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html) for its vector store, but it also provides extensibility points for adding alternative graph stores and vector stores. The default backend for LLMs and embedding models is [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/); but, as with the stores, the toolkit can be configured for other LLM and embedding model backends using LlamaIndex abstractions.

## Installation

The graphrag-toolkit requires python and [pip](http://www.pip-installer.org/en/latest/) to install. You can install the graphrag-toolkit using pip:

```
$ pip install https://github.com/awslabs/graphrag-toolkit/releases/latest/download/graphrag-toolkit.zip
```

### Supported Python versions

The graphrag-toolkit requires Python 3.10 or greater.

## Example of use

### Constructing a graph

```
from graphrag_toolkit import LexicalGraphIndex
from graphrag_toolkit.storage import GraphStoreFactory
from graphrag_toolkit.storage import VectorStoreFactory

from llama_index.readers.web import SimpleWebPageReader

import nest_asyncio
nest_asyncio.apply()

doc_urls = [
    'https://docs.aws.amazon.com/neptune/latest/userguide/intro.html',
    'https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html',
    'https://docs.aws.amazon.com/neptune-analytics/latest/userguide/neptune-analytics-features.html',
    'https://docs.aws.amazon.com/neptune-analytics/latest/userguide/neptune-analytics-vs-neptune-database.html'
]

docs = SimpleWebPageReader(
    html_to_text=True,
    metadata_fn=lambda url:{'url': url}
).load_data(doc_urls)

graph_store = GraphStoreFactory.for_graph_store(
    'neptune-db://my-graph.cluster-abcdefghijkl.us-east-1.neptune.amazonaws.com'
)

vector_store = VectorStoreFactory.for_vector_store(
    'aoss://https://abcdefghijkl.us-east-1.aoss.amazonaws.com'
)

graph_index = LexicalGraphIndex(
    graph_store, 
    vector_store
)

graph_index.extract_and_build(docs)
```

### Querying the graph

```
from graphrag_toolkit import LexicalGraphQueryEngine
from graphrag_toolkit.storage import GraphStoreFactory
from graphrag_toolkit.storage import VectorStoreFactory

import nest_asyncio
nest_asyncio.apply()

graph_store = GraphStoreFactory.for_graph_store(
    'neptune-db://my-graph.cluster-abcdefghijkl.us-east-1.neptune.amazonaws.com'
)

vector_store = VectorStoreFactory.for_vector_store(
    'aoss://https://abcdefghijkl.us-east-1.aoss.amazonaws.com'
)

query_engine = LexicalGraphQueryEngine.for_traversal_based_search(
    graph_store, 
    vector_store
)

response = query_engine.query("What are the differences between Neptune Database and Neptune Analytics?")

print(response.response)
```

## Documentation

  - [Storage Model](./docs/storage-model.md) 
  - [Indexing](./docs/indexing.md) 
  - [Querying](./docs/querying.md) 
  - [Configuration](./docs/configuration.md) 
  - [Graph Model](./docs/graph-model.md)


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

