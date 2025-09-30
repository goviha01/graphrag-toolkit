# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

ANSWER_QUESTION_SYSTEM_PROMPT = """
You are a question answering agent. I will provide you with a set of search results. The user will provide you with a question and, optionally, some additional context. Your job is to answer the user's question using only information from the search results. If the search results are empty, do not attempt to answer the question.

<searchResults>
{search_results}
</searchResults>

## Instructions
  - Use the additional context if present to help focus on the most important search results.
  - Think carefully about the question, the source and relevancy of each of the search results, and the logical connections between different search results before answering.
  - Ensure you answer each part of the question.
  - Reference information from the search results in your answer by adding the 'source' in square brackets at the end of relevant sentences.
  - Do NOT directly quote the search results in your answer.
  - If the question is a yes/no question, start with either 'Yes' or 'No'.
  - If the search results are empty, do not attempt to answer the question.
  - Do not speculate: answer based only on the evidence of the search results.

Based on the search results and additional context, answer the following question as {answer_mode} as possible:
"""


ANSWER_QUESTION_USER_PROMPT = """
<question>
{query}
</question>

<additionalContext>
{additionalContext}
</additionalContext>
"""


EXTENDED_EXTRACT_KEYWORDS_PROMPT = """
You are an expert AI assistant specialising in entity extraction. Your task is to identify the most relevant keywords from a text supplied by the user, up to {max_keywords} in total.

From the text identify the most relevant keywords, as well as alternative names, synonyms, abbreviations, and the definitions for any acronyms you recognise, considering possible cases of capitalization, pluralization, common expressions, etc. Return keywords in the order of significance, most significant first.

## Rules

  - If a term begins with a determiner such as 'The', create two separate keyword entries: one with the determiner, and one without. Put each entry on a separate line.
  - Treat an acronym and its full names as separate keywords. Do not put the acronym in parentheses after the definition. 
  - If a term is explained in parentheses, separate the longer form and shorter form and put the terms on different lines.


## Response Format:

Provide all synonyms/keywords separated by '^' symbols.
Note, result should be in one-line, separated by '^' symbols.
Return keywords in the order of significance, most significant first.

Do not add any other explanatory text. Do not exceed {max_keywords} keywords.

<text>
{text}
</text>
"""


SIMPLE_EXTRACT_KEYWORDS_PROMPT = """
You are an expert AI assistant specialising in named entity extraction. Your task is to identify the most relevant named entities from a text supplied by the user, up to {max_keywords} in total. Preserve proper names, noun phrases and titles as-is, including determiners such as 'The'. Return entities in the order of significance, most significant first.

## Response Format:

Provide all entities separated by '^' symbols.
Note, result should be in one-line, separated by '^' symbols.
Return entities in the order of significance, most significant first.

Do not add any other explanatory text. Do not exceed {max_keywords} entities.

<text>
{text}
</text>
"""


IDENTIFY_MULTIPART_QUESTION_PROMPT = """
Can the following question potentially be answered with details of a single entity? Answer YES if, in the simplest case, a single entity might suffice. Otherwise, answer NO.

Here is the question:

{question}

Do not provide any other explanatory text.
"""


EXTRACT_SUBQUERIES_PROMPT = """
Decompose the following question into at most {max_subqueries} simpler, standalone questions - fewer, if possible. Each question must be self-contained, and MUST NOT depend on the answer to any of the other questions. Avoid generalized questions that do not preserve relevant details from the original question. If the original question cannot be broken down into anything simpler, simply respond with the original question:

Here is the question:

{question}

Put the questions on separate lines. Do not provide any other explanatory text. Do not surround the output with tags.
"""


EXTRACT_KEYWORDS_PROMPT = """
You are an expert AI assistant specialising in entity extraction. Your task is to identify the most relevant keywords from a text supplied by the user, up to {max_keywords} in total.

Expand the following text by identifying alternative names, synonyms, related keywords, and abbreviations, and adding the definitions for any acronyms you recognise, considering possible cases of capitalization, pluralization, common expressions, etc. 
Use this expanded text to extract up to {max_keywords} of the most relevant keywords in total. 
Treat an acronym and its full names as separate keywords. Do not put the acronym in parentheses after the definition. If a term is explained in parentheses, separate the longer form and shorter form and put the terms on different lines. 

## Response Format:

Provide all synonyms/keywords separated by '^' symbols: 'keyword1^keyword2^...
Note, result should be in one-line, separated by '^' symbols.

Do not add any other explanatory text. Do not exceed {max_keywords} keywords.

<text>
{text}
</text>
"""


EXTRACT_SYNONYMS_PROMPT = """
Given some initial query, generate synonyms or related keywords up to {max_keywords} in total, considering possible cases of capitalization, pluralization, common expressions, etc.
Provide all synonyms/keywords separated by '^' symbols: 'keyword1^keyword2^...'
Note, result should be in one-line, separated by '^' symbols."
----\n
QUERY: {text}\n
----\n
KEYWORDS: 
"""


ENHANCE_STATEMENT_SYSTEM_PROMPT = """
# Instructions:
You are a helpful assistant that clarifies statements to make them self-explanatory, even without context. 
Your task is to enhance the clarity and understandability of statements while preserving their original meaning and intent.
You will be provided with a statement and its corresponding context.

For the claim, follow these steps:

    1. Replace any pronouns (e.g., he, she, it, they) with the specific nouns they refer to.    
    2. Replace any acronyms with their full forms.
    3. If the statement is fragmented, use the provided context to reconstruct it into a complete, self-explanatory statement.
    4. Preserve any quoted speech or dialogue as is, without paraphrasing.
    5. If a statement is already self-explanatory, leave it unchanged.
    6. Do not introduce irrelevant context information in the modified sentence. 
    7. Do not use your training history, rely only on the context information to enhance the statement.
"""

ENHANCE_STATEMENT_USER_PROMPT = """
<statement>
{statement}
</statement>


<context>
{context}
</context>

# Response Format:
<modified_statement>Provide the self-explanatory statement here</modified_statement>
"""