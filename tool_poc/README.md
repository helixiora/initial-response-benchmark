# Dataset Tool Calling POC

This project demonstrates how to use OpenAI's tool calling feature to answer questions from specialized datasets. The system loads datasets, creates tools for each dataset, and ensures that responses only contain information from the datasets.

## Datasets

The project includes several example datasets:

- **Pokemon**: Information about Pokemon, including starters, legendary Pokemon, and type effectiveness.
- **Space**: Information about space, including planets, galaxies, and black holes.
- **History**: Information about historical periods, including Ancient Rome, the Renaissance, and the Industrial Revolution.

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

## Usage

### Local File Version

Run the interactive script to ask questions about the datasets:

```
python interactive.py
```

You can also run a set of example questions:

```
python interactive.py --examples
```

Enable verbose mode to see detailed information about the process:

```
python interactive.py --verbose
```

Or combine both flags:

```
python interactive.py --examples --verbose
```

### Vector Database Version

This project also includes a vector database implementation using Pinecone. This approach chunks the documents, creates embeddings, and stores them in a vector database for semantic search.

#### Step 1: Set up the Vector Database

First, run the setup script to process the datasets and create a Pinecone index:

```
python vector_setup.py
```

You can specify a custom index name:

```
python vector_setup.py --index my-custom-index
```

This script will:
1. Load all datasets
2. Chunk the documents using LangChain's RecursiveCharacterTextSplitter
3. Create embeddings using OpenAI's embedding model
4. Store the chunks in a Pinecone index
5. Save the configuration to `vector_config.json`

#### Step 2: Use the Vector Database Interactive Script

Once the vector database is set up, you can use the vector-based interactive script:

```
python vector_interactive.py
```

Just like the file-based version, you can run example questions and enable verbose mode:

```
python vector_interactive.py --examples
python vector_interactive.py --verbose
python vector_interactive.py --examples --verbose
```

## Features

- **Dataset Loading**: Automatically loads datasets from the `datasets` directory.
- **Tool Creation**: Creates tools for each dataset to use with OpenAI's tool calling feature.
- **Validation**: Ensures that responses only contain information from the datasets.
- **Source Tracking**: Shows which file(s) in a dataset contained the answer.
- **Vector Search**: The vector database version uses semantic search to find the most relevant chunks.
- **Colorful Output**: Uses ANSI color codes and Unicode symbols for a better user experience.

## Verbose Mode

Enabling verbose mode provides additional information about the process:

- Dataset loading
- Tool selection
- Validation process
- Timing information
- Vector search details (in the vector database version)

## Adding New Datasets

To add a new dataset:

1. Create a new directory in the `datasets` directory
2. Add a `metadata.json` file with the following structure:
   ```json
   {
     "name": "YourDatasetName",
     "description": "Description of your dataset",
     "files": [
       {
         "filename": "file1.txt",
         "description": "Description of file1"
       },
       {
         "filename": "file2.txt",
         "description": "Description of file2"
       }
     ]
   }
   ```
3. Add the files referenced in the metadata file

## Vector Database vs. Local Files

The project provides two approaches for accessing dataset information:

1. **Local Files** (`interactive.py`): Reads directly from text files on disk. Simple but less scalable.
2. **Vector Database** (`vector_setup.py` and `vector_interactive.py`): Uses Pinecone to store and retrieve chunks based on semantic similarity. More scalable and provides better results for complex queries.

### Advantages of the Vector Database Approach:

- **Semantic Search**: Finds relevant information based on meaning, not just exact matches
- **Chunking**: Breaks documents into smaller pieces for more precise retrieval
- **Scalability**: Can handle much larger datasets efficiently
- **Cross-Document Retrieval**: Can find related information across multiple files

When working with large datasets or complex queries, the vector database approach is recommended. 