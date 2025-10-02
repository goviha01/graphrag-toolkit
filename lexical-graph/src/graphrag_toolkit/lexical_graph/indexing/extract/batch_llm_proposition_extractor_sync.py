# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
import concurrent.futures
import uuid
import json
import shutil
from typing import Optional, Sequence, List, Dict, Any, cast
from datetime import datetime

from graphrag_toolkit.lexical_graph import GraphRAGConfig, BatchJobError
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.indexing.model import Propositions
from graphrag_toolkit.lexical_graph.indexing.constants import PROPOSITIONS_KEY
from graphrag_toolkit.lexical_graph.indexing.prompts import EXTRACT_PROPOSITIONS_PROMPT
from graphrag_toolkit.lexical_graph.indexing.extract.batch_config import BatchConfig
from graphrag_toolkit.lexical_graph.indexing.extract.llm_proposition_extractor import LLMPropositionExtractor

from graphrag_toolkit.lexical_graph.indexing.utils.batch_inference_utils import get_file_size_mb, get_file_sizes_mb, split_nodes, get_request_body, create_and_run_batch_job, download_output_files, process_batch_output_sync
from graphrag_toolkit.lexical_graph.indexing.utils.batch_inference_utils import BEDROCK_MIN_BATCH_SIZE

from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeRelationship

logger = logging.getLogger(__name__)


class BatchLLMPropositionExtractorSync(BaseExtractor):

    batch_config:BatchConfig = Field('Batch inference config')
    llm:Optional[LLMCache] = Field(
        description='The LLM to use for extraction'
    )
    prompt_template:str = Field(description='Prompt template')
    source_metadata_field:Optional[str] = Field(description='Metadata field from which to extract propositions')
    batch_inference_dir:str = Field(description='Directory for batch inputs and outputs')
    

    @classmethod
    def class_name(cls) -> str:
       return 'BatchLLMPropositionExtractorSync'
    
    def __init__(self, 
                 batch_config:BatchConfig,
                 llm:LLMCacheType=None,
                 prompt_template:str = None,
                 source_metadata_field:Optional[str] = None,
                 batch_inference_dir:str = None):
        
        super().__init__(
            batch_config = batch_config,
            llm = llm if llm and isinstance(llm, LLMCache) else LLMCache(
                llm=llm or GraphRAGConfig.extraction_llm,
                enable_cache=GraphRAGConfig.enable_cache
            ),
            prompt_template=prompt_template or EXTRACT_PROPOSITIONS_PROMPT,
            source_metadata_field=source_metadata_field,
            batch_inference_dir=batch_inference_dir or os.path.join('output', 'batch-propositions')
        )

        logger.debug(f'Prompt template: {self.prompt_template}')

        self._prepare_directory(self.batch_inference_dir)

    def _prepare_directory(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        return dir
    
    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        raise NotImplemented()
    
    def _process_single_batch(self, batch_index:int, node_batch:List[TextNode], s3_client, bedrock_client):
        try:

            batch_start = time.time()
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            batch_suffix = f'{batch_index}-{uuid.uuid4().hex[:5]}'
            input_filename = f'proposition-extraction-{timestamp}-batch-{batch_suffix}.jsonl'

            root_dir = os.path.join(self.batch_inference_dir, timestamp, batch_suffix)
            input_dir = os.path.join(root_dir, 'inputs')
            output_dir = os.path.join(root_dir, 'outputs')
            self._prepare_directory(input_dir)
            self._prepare_directory(output_dir)

            input_filepath = os.path.join(input_dir, input_filename)

            logger.debug(f'[Proposition batch inputs] Writing records to {input_filename}')

            llm = self.llm.llm

            inference_parameters = llm._get_all_kwargs() 
            record_count = 0

            with open(input_filepath, 'w') as file:

                for node in node_batch:
                    text = node.metadata.get(self.source_metadata_field, node.text) if self.source_metadata_field else node.text
                    source = node.relationships.get(NodeRelationship.SOURCE, None)
                    if source:
                        source_info = '\n'.join([str(v) for v in source.metadata.values()])
                    else:
                        source_info = ''
                    
                    messages = llm._get_messages(PromptTemplate(self.prompt_template), text=text, source_info=source_info)
                    json_structure = {
                        'recordId': node.node_id,
                        'modelInput': get_request_body(llm, messages, inference_parameters)
                    }

                    json.dump(json_structure, file)
                    file.write('\n')

                    record_count += 1

            logger.debug(f'[Proposition batch inputs] Batch input file ready [num_records: {record_count}, file: {input_filepath} ({get_file_size_mb(input_filepath)} MB)]')

            # 2 - Upload records to s3
            if self.batch_config.key_prefix:
                s3_input_key = os.path.join(self.batch_config.key_prefix, 'batch-propositions', timestamp, batch_suffix, 'inputs', os.path.basename(input_filename))
                s3_output_path = os.path.join(self.batch_config.key_prefix, 'batch-propositions', timestamp, batch_suffix, 'outputs/')
            else:
                s3_input_key = os.path.join('batch-propositions', timestamp, batch_suffix, 'inputs', os.path.basename(input_filename))
                s3_output_path = os.path.join('batch-propositions', timestamp, batch_suffix, 'outputs/')

            upload_start = time.time()
            logger.debug(f'[Proposition batch inputs] Started uploading {input_filename} to S3 [bucket: {self.batch_config.bucket_name}, key: {s3_input_key}]')
            s3_client.upload_file(input_filepath, self.batch_config.bucket_name, s3_input_key)
            upload_end = time.time()
            logger.debug(f'[Proposition batch inputs] Finished uploading {input_filename} to S3 [bucket: {self.batch_config.bucket_name}, key: {s3_input_key}] ({int((upload_end - upload_start) * 1000)} millis)')

            # 3 - Invoke batch job
            create_and_run_batch_job(
                'extract-propositions',
                bedrock_client, 
                timestamp, 
                batch_suffix,
                self.batch_config,
                s3_input_key, 
                s3_output_path,
                self.llm.model
            )

            download_start = time.time()
            logger.debug(f'[Proposition batch outputs] Started downloading outputs to {output_dir} from S3 [bucket: {self.batch_config.bucket_name}, key: {s3_output_path}]')
            download_output_files(s3_client, self.batch_config.bucket_name, s3_output_path, input_filename, output_dir)
            download_end = time.time()
            logger.debug(f'[Proposition batch outputs] Finished downloading outputs to {output_dir} from S3 [bucket: {self.batch_config.bucket_name}, key: {s3_output_path}]  ({int((download_end - download_start) * 1000)} millis)')
            
            output_file_stats = [f'{f} ({size} MB)' for f, size in get_file_sizes_mb(output_dir).items()]
            logger.debug(f'[Proposition batch outputs] Batch output files ready [files: {output_file_stats}]')

            # 4 - Once complete, process batch output
            for (node_id, text) in process_batch_output_sync(output_dir, input_filename, self.llm):
                yield (node_id, text)

            batch_end = time.time()
            logger.debug(f'[Proposition batch outputs] Completed processing of batch {batch_index} ({int(batch_end-batch_start)} seconds)')
            
            if self.batch_config.delete_on_success:
                def log_delete_error(function, path, excinfo):
                    logger.error(f'[Proposition batch] Error deleteing {path} - {str(excinfo[1])}' )

                logger.debug(f'[Proposition batch] Deleting batch directory: {root_dir}' )
                shutil.rmtree(root_dir, onerror=log_delete_error)

        except Exception as e:
            batch_end = time.time()
            raise BatchJobError(f'[Proposition batch] Error processing batch {batch_index} ({int(batch_end-batch_start)} seconds): {str(e)}') from e 
    
    
    def _process_nodes(self, 
        nodes:Sequence[BaseNode],
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
        **kwargs: Any
    ):

        node_metadata_map = {}

        if len(nodes) < BEDROCK_MIN_BATCH_SIZE:

            logger.info(f'[Proposition batch] Not enough records to run batch extraction. List of nodes contains fewer records ({len(nodes)}) than the minimum required by Bedrock ({BEDROCK_MIN_BATCH_SIZE}), so running LLMPropositionExtractor instead.')
            
            extractor = LLMPropositionExtractor(
                prompt_template=self.prompt_template, 
                source_metadata_field=self.source_metadata_field
            )

            node_metadata_map.update(extractor.extract(nodes))

        else:

            s3_client = GraphRAGConfig.s3
            bedrock_client = GraphRAGConfig.bedrock

            # 1 - Split nodes into batches (if needed)
            node_batches = split_nodes(nodes, self.batch_config.max_batch_size)
            logger.debug(f'[Proposition batch] Split nodes into batches [num_batches: {len(node_batches)}, sizes: {[len(b) for b in node_batches]}]')

            # 2 - Process batches concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_config.max_num_concurrent_batches) as executor:

                futures = [
                    executor.submit(self._process_single_batch, batch_index, node_batch, s3_client, bedrock_client)
                    for batch_index, node_batch in enumerate(node_batches)
                ]
                
                executor.shutdown()

                for future in futures:
                    for (node_id, text) in future.result():
                        node_metadata_map[node_id] = text

         # 3 - Process proposition nodes
        for node in nodes:
            
            if node.node_id in node_metadata_map:
                raw_response = node_metadata_map[node.node_id]
                propositions = raw_response.split('\n')
                propositions_model = Propositions(propositions=[p for p in propositions if p])
                node.metadata[PROPOSITIONS_KEY] = propositions_model.model_dump()['propositions']                
            else:
                node.metadata[PROPOSITIONS_KEY] = []

            if excluded_embed_metadata_keys is not None:
                node.excluded_embed_metadata_keys.extend(excluded_embed_metadata_keys)
            if excluded_llm_metadata_keys is not None:
                node.excluded_llm_metadata_keys.extend(excluded_llm_metadata_keys)
            if not self.disable_template_rewrite:
                if isinstance(node, TextNode):
                    cast(TextNode, node).text_template = self.node_text_template

        return nodes
    
    def __call__(self, 
        nodes: Sequence[BaseNode],
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        return self._process_nodes(nodes, excluded_embed_metadata_keys, excluded_llm_metadata_keys, **kwargs)
    
