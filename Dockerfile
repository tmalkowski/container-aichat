# Dockerfile
FROM langflowai/langflow:latest

RUN mkdir -p /app/components
COPY opensearch_langflow_tools.py /app/components/

RUN pip install opensearch-py sentence-transformers
