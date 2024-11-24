"""
This script processes a text file, generates embeddings using a huggingface embedding model,
stores them in a Qdrant vector database, performs a search based on a provided
query, and generates an answer using a huggingface chat LLM. It accepts command-line
arguments for the search query (-q or --query) and the context text file name
(-cf or --context-file). The generated answer is written to an output text file.
"""

import os
import sys
import time
import uuid
import argparse
# Removed: from rich.console import Console
from langchain_text_splitters import RecursiveCharacterTextSplitter
import llama_cpp
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def chunk(iterable, size):
    """Yield successive chunks from iterable of given size."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def read_and_split_text(file_path):
    """Read text from a file and split it into documents."""
    with open(file_path, 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.create_documents([text])
    return documents

def create_document_embeddings(documents, llm):
    """Create embeddings for a list of documents using the provided language model."""
    batch_size = 100
    documents_embeddings = []
    batches = list(chunk(documents, batch_size))
    start = time.time()
    for batch in batches:
        embeddings = llm.create_embedding([item.page_content for item in batch])
        documents_embeddings.extend(
            [
                (document, embedding['embedding'])
                for document, embedding in zip(batch, embeddings['data'])
            ]
        )
    end = time.time()
    all_text = [item.page_content for item in documents]
    char_per_sec = len(''.join(all_text)) / (end - start)
    print(f"Time: {end - start:.2f} seconds / {char_per_sec:,.2f} chars/sec")
    return documents_embeddings

def store_embeddings_in_qdrant(documents_embeddings,
                               collection_name="zwanzig-book",
                               vector_size=1024):
    """Store document embeddings in a Qdrant vector database."""
    client = QdrantClient(path="embeddings")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": doc.page_content},
        )
        for doc, embedding in documents_embeddings
    ]
    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )
    return client

def search_embeddings(client, search_query, llm, collection_name="zwanzig-book", limit=6):
    """Search the Qdrant vector database using the query's embedding."""
    query_vector = llm.create_embedding([search_query])['data'][0]['embedding']
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
    )
    return search_result

def generate_answer(search_result, search_query, llm):
    """Generate an answer using the language model and search results."""
    template = """You are a helpful assistant who answers questions using only the provided context.
If you don't know the answer, simply state that you don't know.

{context}

Question: {question}"""

    context = "\n\n".join([row.payload['text'] for row in search_result])
    prompt = template.format(context=context, question=search_query)
    stream = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    response = ''
    for response_chunk in stream:
        content = response_chunk['choices'][0]['delta'].get('content', '')
        response += content
    return response

def main():
    """Main function to execute the script."""
    parser = argparse.ArgumentParser(
        description='Process a text file and answer a query using local language models.'
    )
    parser.add_argument('-q', '--query', required=True, help='The search query')
    parser.add_argument('-cf', '--context-file', required=True, help='The context text file name')
    parser.add_argument('-of', '--output-file', default='output.txt', help='The output text file name')

    args = parser.parse_args()

    search_query = args.query
    file_path = args.context_file
    output_file = args.output_file

    # Removed: console = Console()

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)

    # Step 1: Read and split text
    documents = read_and_split_text(file_path)

    if not documents:
        print(f"Error: No documents were created from the file '{file_path}'.")
        sys.exit(1)

    # Step 2: Load embedding model and create embeddings
    embedding_llm = llama_cpp.Llama.from_pretrained(
        repo_id="mixedbread-ai/mxbai-embed-large-v1",
        filename="gguf/mxbai-embed-large-v1-f16.gguf",
        embedding=True,
    )
    documents_embeddings = create_document_embeddings(documents, embedding_llm)

    # Step 3: Store embeddings in Qdrant
    client = store_embeddings_in_qdrant(documents_embeddings)

    # Step 4: Search embeddings
    search_result = search_embeddings(client, search_query, embedding_llm)

    if not search_result:
        print("No search results found.")
        sys.exit(1)

    # Step 5: Load language model for generating answers
    answer_llm = llama_cpp.Llama.from_pretrained(
        repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
        filename="llama-3.2-3b-instruct-q4_k_m.gguf",
    )

    # Step 6: Generate the answer
    answer = generate_answer(search_result, search_query, answer_llm)

    # Step 7: Write the answer to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(answer)

    print(f"Answer written to {output_file}")

if __name__ == "__main__":
    main()
