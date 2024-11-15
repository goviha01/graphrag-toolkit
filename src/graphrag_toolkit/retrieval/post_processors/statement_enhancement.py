# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
import sys
import logging

from pydantic import Field
from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.async_utils import run_async_tasks

from graphrag_toolkit import GraphRAGConfig
from graphrag_toolkit.retrieval.prompts import ENHANCE_STATEMENT_SYSTEM_PROMPT, ENHANCE_STATEMENT_USER_PROMPT

logger = logging.getLogger(__name__)

class StatementEnhancementPostProcessor(BaseNodePostprocessor):
    """Enhances statements using chunk context while preserving original values."""

    llm: Optional[LLM] = Field(default=None)
    max_concurrent: int = Field(default=10)
    system_prompt: str = Field(default=ENHANCE_STATEMENT_SYSTEM_PROMPT)
    user_prompt: str = Field(default=ENHANCE_STATEMENT_USER_PROMPT)
    enhance_template: ChatPromptTemplate = Field(default=None)

    def __init__(
        self,
        llm=None,
        system_prompt: str = ENHANCE_STATEMENT_SYSTEM_PROMPT,
        user_prompt: str = ENHANCE_STATEMENT_USER_PROMPT,
        max_concurrent: int = 10
    ) -> None:
        super().__init__()
        self.llm = llm or GraphRAGConfig.response_llm
        self.max_concurrent = max_concurrent
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        
        self.enhance_template = ChatPromptTemplate(message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ])

    async def enhance_statement(self, node: NodeWithScore) -> NodeWithScore:
        """Enhance a single statement using its chunk context."""
        try:
            def blocking_llm_call():
                return self.llm.predict(
                    prompt=self.enhance_template,
                    statement=node.node.metadata['statement']['value'],
                    context=node.node.metadata['chunk']['value'],
                )
            
            response = await asyncio.to_thread(blocking_llm_call)
            pattern = r'<modified_statement>(.*?)</modified_statement>'
            match = re.search(pattern, response, re.DOTALL)
            
            if match:
                enhanced_text = match.group(1).strip()
                new_node = TextNode(
                    text=enhanced_text,  
                    metadata={
                        'statement': node.node.metadata['statement'], 
                        'chunk': node.node.metadata['chunk'],
                        'source': node.node.metadata['source'],
                        'search_type': node.node.metadata.get('search_type')
                    }
                )
                return NodeWithScore(node=new_node, score=node.score)
            
            return node
            
        except Exception as e:
            logger.error(f"Error enhancing statement: {e}")
            return node

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Process all nodes with concurrent enhancement."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def enhance_with_semaphore(node: NodeWithScore) -> NodeWithScore:
            async with semaphore:
                return await self.enhance_statement(node)

        enhanced_nodes = run_async_tasks([
            enhance_with_semaphore(node) for node in nodes
        ])
        
        return enhanced_nodes