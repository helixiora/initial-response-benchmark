# Vector Database Tool Calling POC

This project demonstrates how to use OpenAI's tool calling capabilities with a Pinecone vector database to create a system that can answer questions based on specialized datasets.

## Datasets

The system dynamically loads datasets from the `datasets/` directory. Each dataset is a collection of text files that contain information on a specific topic.

To add a new dataset:
1. Create a new folder in the `datasets/` directory (e.g., `datasets/newdataset/`)
2. Add text files (.txt or .md) to the folder with relevant information
3. Run the setup script to process the new dataset

## Setup

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

3. Run the setup script to process datasets and create the vector database:
```
python vector_setup.py
```

This will:
- Scan the `datasets/` directory for available datasets
- Generate descriptions for each dataset using GPT-4o-mini
- Chunk the text files and create embeddings
- Upload the embeddings to a Pinecone vector database
- Save the dataset information and configuration for the interactive script

## Usage

Run the interactive script to query the datasets:
```
python vector_interactive.py
```

You can also run example questions for all datasets:
```
python vector_interactive.py --examples
```

### Verbose Mode

Enable verbose mode to see detailed information about the process:
```
python vector_interactive.py --verbose
```

Or with example questions:
```
python vector_interactive.py --examples --verbose
```

Verbose mode provides insights into:
- Dataset loading
- Tool selection
- Vector search results
- Validation process
- Timing information

## How It Works

1. The system loads dataset information and creates tool definitions for each dataset
2. When a user asks a question, GPT-4o-mini determines which dataset tool to use
3. The system performs a vector search in Pinecone to find relevant information
4. GPT-4o-mini generates an answer based strictly on the retrieved information
5. A validation step ensures the answer only contains information from the dataset

## Model

This system uses the GPT-4o-mini model for all operations, including:
- Determining which dataset to use
- Generating final answers
- Validating responses
- Creating dataset descriptions

## Robust JSON Handling

The system includes robust JSON parsing for LLM-generated responses:

- **Trailing Comma Removal**: Automatically removes trailing commas that would make JSON invalid
- **Code Block Extraction**: Extracts JSON from markdown code blocks
- **Object Extraction**: Identifies and extracts JSON objects from surrounding text
- **Error Handling**: Provides detailed error messages and fallbacks when JSON parsing fails
- **Structured Output**: Instructs the LLM to return properly formatted JSON

This ensures reliable operation even when the LLM doesn't produce perfectly formatted JSON.

## Adding New Datasets

To add a new dataset:
1. Create a new folder in the `datasets/` directory
2. Add text files (.txt or .md) with relevant information
3. Run the setup script again to process the new dataset:
```
python vector_setup.py
```

The system will automatically:
- Detect the new dataset
- Generate appropriate descriptions
- Create embeddings and store them in Pinecone
- Make the dataset available in the interactive script 