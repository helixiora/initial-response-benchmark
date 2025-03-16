#!/usr/bin/env python3
"""
Script to set up Pinecone vector database for dataset tool calling.
This script processes the datasets, chunks them, creates embeddings, and stores them in Pinecone.
"""

import os
import json
import time
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import re

# Load environment variables
load_dotenv()

# Check for API keys
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please create a .env file based on .env.example and add your API key.")
    exit(1)

if not os.getenv("PINECONE_API_KEY"):
    print("Error: PINECONE_API_KEY not found in environment variables.")
    print("Please add your Pinecone API key to your .env file.")
    exit(1)

# Initialize clients
openai_client = OpenAI()
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# Unicode symbols for terminal output
class Symbols:
    INFO = "ℹ️ "
    SUCCESS = "✅ "
    WARNING = "⚠️ "
    ERROR = "❌ "
    ROCKET = "🚀 "
    DATABASE = "🗄️ "
    SEARCH = "🔍 "
    FILE = "📄 "
    FOLDER = "📁 "
    PROCESSING = "⚙️ "
    EMBEDDING = "🧠 "
    CLOCK = "🕒 "


def scan_datasets_directory(datasets_dir: str) -> List[Dict[str, Any]]:
    """Scan the datasets directory and discover available datasets."""
    datasets = []

    # Get all subdirectories in the datasets directory
    dataset_dirs = [
        d
        for d in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, d))
    ]

    for dataset_dir in dataset_dirs:
        dataset_path = os.path.join(datasets_dir, dataset_dir)

        # Get all text files in the dataset directory
        dataset_files = []
        for filename in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, filename)
            if os.path.isfile(file_path) and filename.endswith((".txt", ".md")):
                dataset_files.append(filename)

        if dataset_files:  # Only add if there are valid files
            # Generate dataset info from directory name
            dataset_name = dataset_dir.capitalize()

            # Create placeholder descriptions that will be replaced later
            placeholder_description = f"[PLACEHOLDER] Information about {dataset_name}"
            placeholder_tool_description = (
                f"[PLACEHOLDER] Use this tool to answer questions about {dataset_name}"
            )

            # Add dataset to the list
            datasets.append(
                {
                    "name": dataset_name,
                    "description": placeholder_description,
                    "tool_description": placeholder_tool_description,
                    "path": dataset_path,
                    "files": dataset_files,
                }
            )

    return datasets


def get_dataset_content(dataset_path: str, files: List[str]) -> Dict[str, str]:
    """Get the content of all files in a dataset."""
    results = {}

    for filename in files:
        file_path = os.path.join(dataset_path, filename)

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                results[filename] = content

    return results


def generate_dataset_description(
    dataset_name: str, content_dict: Dict[str, str]
) -> Dict[str, str]:
    """Generate a more detailed description of the dataset using OpenAI."""
    # If content dictionary is empty, return generic descriptions
    if not content_dict:
        return {
            "description": f"Information about {dataset_name}",
            "tool_description": f"Use this tool to answer questions about {dataset_name}",
        }

    # Combine a sample of content from each file (first 500 chars)
    sample_content = ""
    for filename, content in content_dict.items():
        sample_content += f"\nExcerpt from {filename}:\n{content[:500]}...\n"

    # Create a prompt for OpenAI to generate descriptions
    prompt = f"""
Based on the following content samples from a dataset named '{dataset_name}', 
please generate two descriptions:
1. A concise description of what this dataset contains (1-2 sentences)
2. A tool description that explains when to use this dataset for answering questions (1-2 sentences)

Content samples:
{sample_content}

Please format your response as a JSON object with 'description' and 'tool_description' fields.
"""

    # Define the response structure
    response_format = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "A concise description of what this dataset contains (1-2 sentences)",
            },
            "tool_description": {
                "type": "string",
                "description": "A tool description that explains when to use this dataset for answering questions (1-2 sentences)",
            },
        },
        "required": ["description", "tool_description"],
    }

    # Call OpenAI to generate descriptions with structured output
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "description": result.get(
                "description", f"Information about {dataset_name}"
            ),
            "tool_description": result.get(
                "tool_description",
                f"Use this tool to answer questions about {dataset_name}",
            ),
        }
    except Exception as e:
        print(
            f"{Symbols.ERROR}{Colors.RED}Error generating description: {str(e)}{Colors.ENDC}"
        )

    # Fallback to generic descriptions
    return {
        "description": f"Information about {dataset_name}",
        "tool_description": f"Use this tool to answer questions about {dataset_name}",
    }


def chunk_text(text: str, filename: str, dataset_name: str) -> List[Dict[str, Any]]:
    """Chunk text using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_text(text)

    # Create document objects with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        # Create a deterministic ID based on content
        chunk_id = hashlib.md5(
            f"{dataset_name}_{filename}_{i}_{chunk[:50]}".encode()
        ).hexdigest()

        documents.append(
            {
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "dataset": dataset_name,
                    "filename": filename,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                },
            }
        )

    return documents


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings using OpenAI's embedding model."""
    response = openai_client.embeddings.create(
        input=texts, model="text-embedding-3-small"
    )

    return [data.embedding for data in response.data]


def setup_pinecone_index(index_name: str, dimension: int = 1536) -> None:
    """Set up a Pinecone index if it doesn't exist."""
    # Check if index already exists
    existing_indexes = [index.name for index in pinecone_client.list_indexes()]

    if index_name in existing_indexes:
        print(
            f"{Symbols.INFO}{Colors.YELLOW}Index '{index_name}' already exists. Using existing index.{Colors.ENDC}"
        )
        return

    # Create a new index
    print(
        f"{Symbols.DATABASE}{Colors.BLUE}Creating new Pinecone index: '{index_name}'...{Colors.ENDC}"
    )

    pinecone_client.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )

    # Wait for index to be ready
    print(
        f"{Symbols.CLOCK}{Colors.YELLOW}Waiting for index to initialize...{Colors.ENDC}"
    )
    time.sleep(30)  # Give it some time to initialize

    print(
        f"{Symbols.SUCCESS}{Colors.GREEN}Index '{index_name}' created successfully.{Colors.ENDC}"
    )


def is_placeholder_description(description: str) -> bool:
    """Check if a description is a placeholder that needs to be updated."""
    return description.startswith("[PLACEHOLDER]")


def process_dataset(
    dataset: Dict[str, Any], index_name: str, batch_size: int = 100
) -> None:
    """Process a dataset, create embeddings, and upload to Pinecone."""
    dataset_name = dataset["name"]
    print(
        f"\n{Symbols.PROCESSING} {Colors.BOLD}Processing dataset: {dataset_name}{Colors.ENDC}"
    )

    # Get dataset content
    content_dict = get_dataset_content(dataset["path"], dataset["files"])

    # Generate descriptions if they're placeholders or if we have content
    if is_placeholder_description(dataset["description"]) or content_dict:
        descriptions = generate_dataset_description(dataset_name, content_dict)
        dataset["description"] = descriptions["description"]
        dataset["tool_description"] = descriptions["tool_description"]
        print(
            f"{Symbols.INFO}{Colors.CYAN}Generated description: {dataset['description']}{Colors.ENDC}"
        )
    else:
        print(
            f"{Symbols.INFO}{Colors.CYAN}Using existing description: {dataset['description']}{Colors.ENDC}"
        )

    # Chunk the documents
    all_chunks = []
    for filename, content in content_dict.items():
        print(f"{Symbols.FILE}{Colors.BLUE}Chunking file: {filename}{Colors.ENDC}")
        chunks = chunk_text(content, filename, dataset_name)
        all_chunks.extend(chunks)

    print(
        f"{Symbols.INFO}{Colors.CYAN}Created {len(all_chunks)} chunks from {len(content_dict)} files{Colors.ENDC}"
    )

    # Get the Pinecone index
    index = pinecone_client.Index(index_name)

    # Process in batches to avoid rate limits
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in batch]

        print(
            f"{Symbols.EMBEDDING}{Colors.YELLOW}Creating embeddings for batch {i // batch_size + 1}/{(len(all_chunks) - 1) // batch_size + 1}...{Colors.ENDC}"
        )
        embeddings = create_embeddings(texts)

        # Prepare vectors for Pinecone
        vectors = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            vectors.append(
                {
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": {
                        "text": chunk["text"],
                        "dataset": chunk["metadata"]["dataset"],
                        "filename": chunk["metadata"]["filename"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                        "chunk_count": chunk["metadata"]["chunk_count"],
                    },
                }
            )

        # Upsert to Pinecone
        print(
            f"{Symbols.DATABASE}{Colors.BLUE}Uploading batch to Pinecone...{Colors.ENDC}"
        )
        index.upsert(vectors=vectors)

        # Avoid rate limits
        if i + batch_size < len(all_chunks):
            print(
                f"{Symbols.CLOCK}{Colors.YELLOW}Pausing for 2 seconds to avoid rate limits...{Colors.ENDC}"
            )
            time.sleep(2)

    print(
        f"{Symbols.SUCCESS}{Colors.GREEN}Successfully processed dataset: {dataset_name}{Colors.ENDC}"
    )


def save_dataset_info(datasets: List[Dict[str, Any]]) -> None:
    """Save dataset information to a JSON file."""
    # Extract only the necessary information
    dataset_info = []
    for dataset in datasets:
        dataset_info.append(
            {
                "name": dataset["name"],
                "description": dataset["description"],
                "tool_description": dataset["tool_description"],
            }
        )

    # Save to file
    dataset_info_path = os.path.join(os.path.dirname(__file__), "dataset_info.json")
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(
        f"{Symbols.SUCCESS}{Colors.GREEN}Dataset information saved to dataset_info.json{Colors.ENDC}"
    )


def main(index_name: Optional[str] = None):
    """Main function to set up Pinecone database for all datasets."""
    print(
        f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 20} Setting Up Pinecone Vector Database {'=' * 20}{Colors.ENDC}"
    )

    # Generate a default index name if not provided
    if not index_name:
        index_name = f"dataset-tools-{int(time.time())}"

    # Load datasets
    datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
    datasets = scan_datasets_directory(datasets_dir)

    print(f"{Symbols.INFO}{Colors.BOLD}Found {len(datasets)} datasets:{Colors.ENDC}")
    for dataset in datasets:
        print(
            f"  - {Colors.CYAN}{dataset['name']}{Colors.ENDC}: {dataset['description']}"
        )

    # Set up Pinecone index
    setup_pinecone_index(index_name)

    # Process each dataset
    for dataset in datasets:
        process_dataset(dataset, index_name)

    # Save dataset information to a JSON file
    save_dataset_info(datasets)

    # Save index name to a config file for the interactive script to use
    config = {
        "pinecone_index": index_name,
        "embedding_model": "text-embedding-3-small",
        "created_at": time.time(),
    }

    config_path = os.path.join(os.path.dirname(__file__), "vector_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"\n{Symbols.SUCCESS}{Colors.GREEN}Vector database setup complete!{Colors.ENDC}"
    )
    print(
        f"{Symbols.INFO}{Colors.CYAN}Index name '{index_name}' saved to vector_config.json{Colors.ENDC}"
    )
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 76}{Colors.ENDC}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up Pinecone vector database for datasets"
    )
    parser.add_argument(
        "--index", type=str, help="Name for the Pinecone index (optional)"
    )
    args = parser.parse_args()

    main(args.index)
