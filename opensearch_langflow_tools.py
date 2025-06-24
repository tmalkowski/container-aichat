#!/usr/bin/env python3

# opensearch_langflow_tools.py

from typing import List, Optional
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from langflow.custom import Component
from langflow.inputs import MessageTextInput, StrInput
from langflow.io import Output
from langflow.schema import Data, Message
import uuid
import time
import os
import json

class OpenSearchClientComponent(Component):
    display_name = "OpenSearch Project Tool"
    description = "Search, insert, and update documents in OpenSearch using session tags and vector similarity."
    icon = "database"

    inputs = [
        MessageTextInput(
            name="query",
            display_name="Query Text",
            tool_mode=True,
        ),
        StrInput(
            name="tags",
            display_name="Comma-separated Tags",
            placeholder="projects, scriptname",
            tool_mode=True,
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            tool_mode=True,
            required=False
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message", method="search_and_update"),
    ]

    def __init__(self):
        super().__init__()
        self.os_url = os.environ.get("OPENSEARCH_URL", "http://10.2.2.5:9200")
        self.index = os.environ.get("OPENSEARCH_INDEX", "langflow-projects")
        self.client = OpenSearch([self.os_url])
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def search_and_update(self) -> Message:
        tags = [tag.strip() for tag in self.tags.split(",") if tag.strip()]
        query_text = self.query
        query_vector = self.embedder.encode(query_text).tolist()

        search_body = {
            "size": 3,
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"tags": tags}},
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": 3
                                }
                            }
                        }
                    ]
                }
            }
        }

        response = self.client.search(index=self.index, body=search_body)
        hits = response.get("hits", {}).get("hits", [])

        if not hits:
            # Insert a new document
            new_id = str(uuid.uuid4())
            doc = {
                "id": new_id,
                "tags": tags,
                "content": query_text,
                "embedding": query_vector,
                "created": time.time(),
                "updated": time.time(),
                "session": self.session_id,
            }
            self.client.index(index=self.index, id=new_id, body=doc)
            reply = f"No matches found. Created a new project document with tags: {tags}"
        else:
            # Assume first match and update it
            doc_id = hits[0]["_id"]
            updated_text = hits[0]["_source"].get("content", "") + "\n\n" + query_text
            self.client.update(index=self.index, id=doc_id, body={
                "doc": {
                    "content": updated_text,
                    "updated": time.time(),
                    "embedding": self.embedder.encode(updated_text).tolist(),
                }
            })
            reply = f"Found and updated document {doc_id} with new content."

        return Message(text=reply)

# If using Langflow in dev mode:
# Drop this in `components/opensearch_langflow_tools.py` and restart.
# Ensure sentence-transformers and opensearch-py are in your env:
# pip install opensearch-py sentence-transformers
