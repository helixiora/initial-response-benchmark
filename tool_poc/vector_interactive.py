#!/usr/bin/env python3
"""
Interactive script to demonstrate dataset tool calling with Pinecone vector database.
This directly uses the OpenAI API to implement tool calling and Pinecone for vector search.
"""

import os
import json
import time
import sys
import argparse
import re
import readline
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from typing import List, Dict, Any, Tuple, Set

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


# ANSI color codes
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


# Unicode symbols
class Symbols:
    INFO = "ℹ️ "
    SUCCESS = "✅ "
    WARNING = "⚠️ "
    ERROR = "❌ "
    ROCKET = "🚀 "
    HOURGLASS = "⏳ "
    SEARCH = "🔍 "
    BOOK = "📚 "
    STAR = "⭐ "
    QUESTION = "❓ "
    THINKING = "🤔 "
    LIGHT_BULB = "💡 "
    CLOCK = "🕒 "
    CHECK = "✓ "
    CROSS = "✗ "
    FILE = "📄 "
    FOLDER = "📁 "
    DATABASE = "🗄️ "
    EMBEDDING = "🧠 "


def load_vector_config() -> Dict[str, Any]:
    """Load vector database configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "vector_config.json")

    if not os.path.exists(config_path):
        print(
            f"{Symbols.ERROR}{Colors.RED}Vector config file not found. Please run vector_setup.py first.{Colors.ENDC}"
        )
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def load_dataset_info() -> List[Dict[str, Any]]:
    """Load dataset information from the JSON file."""
    info_path = os.path.join(os.path.dirname(__file__), "dataset_info.json")

    if not os.path.exists(info_path):
        print(
            f"{Symbols.ERROR}{Colors.RED}Dataset info file not found. Please run vector_setup.py first.{Colors.ENDC}"
        )
        exit(1)

    with open(info_path, "r") as f:
        dataset_info = json.load(f)

    return dataset_info


def create_tools(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create tool definitions for OpenAI API using dataset info."""
    tools = []

    for dataset in datasets:
        tool = {
            "type": "function",
            "function": {
                "name": f"get_{dataset['name'].lower()}_info",
                "description": dataset["tool_description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"The question about {dataset['name']} to answer",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

        tools.append(tool)

    return tools


def get_dataset_symbol(dataset_name: str) -> str:
    """Get the appropriate symbol for a dataset."""
    # Use a generic book symbol for all datasets
    return Symbols.BOOK


def create_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Create an embedding for a single text using OpenAI's embedding model."""
    response = openai_client.embeddings.create(input=[text], model=model)

    return response.data[0].embedding


def vector_search(
    query: str, index_name: str, dataset_filter: str = None, top_k: int = 10
) -> List[Dict[str, Any]]:
    """Search for relevant chunks in the vector database."""
    # Create embedding for the query
    query_embedding = create_embedding(query)

    # Get the Pinecone index
    index = pinecone_client.Index(index_name)

    # Prepare filter if dataset is specified
    filter_dict = None
    if dataset_filter:
        filter_dict = {"dataset": {"$eq": dataset_filter}}

    # Perform the search
    results = index.query(
        vector=query_embedding, top_k=top_k, include_metadata=True, filter=filter_dict
    )

    return results.matches


def format_vector_results(results: List[Dict[str, Any]]) -> Tuple[str, Set[str]]:
    """Format vector search results for the API and track source files."""
    formatted_text = []
    source_files = set()

    # Group chunks by filename for better context
    chunks_by_file = {}
    for match in results:
        filename = match["metadata"]["filename"]
        if filename not in chunks_by_file:
            chunks_by_file[filename] = []

        # Add the chunk with its score and index
        chunks_by_file[filename].append(
            {
                "text": match["metadata"]["text"],
                "score": match["score"],
                "chunk_index": match["metadata"]["chunk_index"],
            }
        )

        # Add to source files
        source_files.add(filename)

    # Format each file's chunks
    for filename, chunks in chunks_by_file.items():
        # Sort chunks by their original index to maintain document flow
        chunks.sort(key=lambda x: x["chunk_index"])

        # Add file header
        formatted_text.append(f"\nContent from {filename}:")

        # Add each chunk
        for chunk in chunks:
            formatted_text.append(f"{chunk['text']}")

    return "\n".join(formatted_text), source_files


def safe_json_parse(json_str: str) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM responses, handling common issues like trailing commas.

    Args:
        json_str: The string containing JSON to parse

    Returns:
        Parsed JSON as a dictionary
    """
    # Try to extract JSON if it's wrapped in markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_str, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)

    # Clean up common JSON issues
    # 1. Remove trailing commas in objects and arrays
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    # 2. Try to extract just the JSON object if there's text around it
    json_object_match = re.search(r"({.*})", json_str, re.DOTALL)
    if json_object_match:
        json_str = json_object_match.group(1)

    # Parse the JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(
            f"{Symbols.WARNING}{Colors.YELLOW}JSON parsing error: {str(e)}{Colors.ENDC}"
        )
        print(f"{Symbols.INFO}{Colors.CYAN}Problematic JSON: {json_str}{Colors.ENDC}")
        # Return empty dict as fallback
        return {}


def process_query(query: str, model: str = "gpt-4o-mini", verbose: bool = False):
    """Process a user query using dataset tools and vector search."""
    # Load vector database config
    config = load_vector_config()
    index_name = config["pinecone_index"]
    embedding_model = config["embedding_model"]

    # Load dataset information
    datasets = load_dataset_info()

    # Create tools
    tools = create_tools(datasets)

    if verbose:
        # Print available datasets
        print(
            f"{Symbols.INFO}{Colors.BOLD}Loaded {len(datasets)} dataset tools:{Colors.ENDC}"
        )
        for dataset in datasets:
            symbol = get_dataset_symbol(dataset["name"])
            print(
                f"{symbol} {Colors.CYAN}{dataset['name']}{Colors.ENDC}: {dataset['description']}"
            )
        print(
            f"{Symbols.DATABASE}{Colors.CYAN}Using Pinecone index: {index_name}{Colors.ENDC}"
        )

    # Start timing
    start_time = time.time()

    # Step 1: Ask the model which tool to use
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to specialized datasets as tools. "
                "Your primary goal is to answer user questions by using the appropriate dataset tool. "
                "IMPORTANT: You must ALWAYS use a tool to answer questions. Do not try to answer from your own knowledge. "
                "When a user asks a question, determine which dataset is most relevant, then use that tool. "
                "Be thorough but concise in your responses."
            ),
        },
        {"role": "user", "content": query},
    ]

    if verbose:
        print(
            f"{Symbols.THINKING}{Colors.YELLOW}Asking model which tool to use...{Colors.ENDC}"
        )

    # Always use gpt-4o-mini regardless of the model parameter
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
    )

    tool_calls = response.choices[0].message.tool_calls

    if not tool_calls:
        elapsed_time = time.time() - start_time
        if verbose:
            print(
                f"{Symbols.ERROR}{Colors.RED}No tool calls made. Elapsed time: {elapsed_time:.2f} seconds{Colors.ENDC}"
            )
        return (
            f"{Symbols.ERROR} I couldn't determine which dataset to use for your question. Please try asking about one of our available datasets: {', '.join([d['name'] for d in datasets])}.",
            set(),
        )

    # Step 2: Execute the tool call
    tool_call = tool_calls[0]
    function_name = tool_call.function.name

    # Safely parse the function arguments
    try:
        function_args = safe_json_parse(tool_call.function.arguments)
    except Exception as e:
        if verbose:
            print(
                f"{Symbols.ERROR}{Colors.RED}Error parsing function arguments: {str(e)}{Colors.ENDC}"
            )
            print(
                f"{Symbols.INFO}{Colors.CYAN}Raw arguments: {tool_call.function.arguments}{Colors.ENDC}"
            )
        function_args = {}

    # Find the dataset
    dataset_name = function_name.split("_")[1].capitalize()
    dataset = next(
        (d for d in datasets if d["name"].lower() == dataset_name.lower()), None
    )

    if verbose:
        symbol = get_dataset_symbol(dataset_name)
        print(f"{symbol} {Colors.GREEN}Using dataset: {dataset_name}{Colors.ENDC}")

    if not dataset:
        elapsed_time = time.time() - start_time
        if verbose:
            print(
                f"{Symbols.ERROR}{Colors.RED}Dataset not found. Elapsed time: {elapsed_time:.2f} seconds{Colors.ENDC}"
            )
        return f"{Symbols.ERROR} Dataset {dataset_name} not found.", set()

    # Step 3: Perform vector search
    if verbose:
        print(
            f"{Symbols.EMBEDDING}{Colors.BLUE}Creating embedding for query...{Colors.ENDC}"
        )
        print(
            f"{Symbols.SEARCH}{Colors.BLUE}Searching Pinecone for relevant information...{Colors.ENDC}"
        )

    # Get the user's actual query from the function arguments
    user_query = function_args.get("query", query)

    # Search the vector database
    search_results = vector_search(
        query=user_query,
        index_name=index_name,
        dataset_filter=dataset_name,
        top_k=15,  # Retrieve more chunks for better context
    )

    if verbose:
        print(
            f"{Symbols.SUCCESS}{Colors.GREEN}Found {len(search_results)} relevant chunks.{Colors.ENDC}"
        )

    # Format the results
    dataset_content, source_files = format_vector_results(search_results)

    if not dataset_content.strip():
        elapsed_time = time.time() - start_time
        if verbose:
            print(
                f"{Symbols.ERROR}{Colors.RED}No relevant content found. Elapsed time: {elapsed_time:.2f} seconds{Colors.ENDC}"
            )
        return (
            f"{Symbols.ERROR} I couldn't find any relevant information in the {dataset_name} dataset for your question.",
            set(),
        )

    # Step 4: Get the final answer with strict instructions
    if verbose:
        print(
            f"{Symbols.LIGHT_BULB}{Colors.YELLOW}Generating final answer...{Colors.ENDC}"
        )

    messages.append(response.choices[0].message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": dataset_content,
        }
    )

    # Add a system message to reinforce the constraint
    messages.append(
        {
            "role": "system",
            "content": (
                "CRITICAL INSTRUCTION: Your response must be based EXCLUSIVELY on the information provided in the dataset. "
                "Do NOT include any information from your training data or general knowledge. "
                "If the dataset doesn't contain enough information to fully answer the question, state this explicitly "
                "and only provide what can be directly supported by the dataset content. "
                "Do not invent or add details that aren't present in the dataset."
            ),
        }
    )

    final_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,  # Use lower temperature for more deterministic responses
    )

    # Step 5: Validate the response against the dataset content
    answer = final_response.choices[0].message.content

    if verbose:
        print(
            f"{Symbols.CHECK}{Colors.BLUE}Validating answer against dataset content...{Colors.ENDC}"
        )

    # Add a validation step to check if the answer contains information not in the dataset
    validation_messages = [
        {
            "role": "system",
            "content": (
                "You are a validation assistant. Your task is to verify if the provided answer "
                "contains ONLY information that is present in the dataset content. "
                "If the answer includes any information not found in the dataset, identify those parts. "
                "Return ONLY 'VALID' if the answer is fully supported by the dataset, "
                "or 'INVALID: [explanation]' if it contains unsupported information."
            ),
        },
        {
            "role": "user",
            "content": f"Dataset content:\n{dataset_content}\n\nAnswer to validate:\n{answer}",
        },
    ]

    validation_response = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=validation_messages, temperature=0.0
    )

    validation_result = validation_response.choices[0].message.content

    # If the validation fails, regenerate the answer with stricter constraints
    if not validation_result.startswith("VALID"):
        if verbose:
            print(
                f"{Symbols.WARNING}{Colors.RED}Validation failed: {validation_result}{Colors.ENDC}"
            )
            print(
                f"{Symbols.THINKING}{Colors.YELLOW}Regenerating answer with stricter constraints...{Colors.ENDC}"
            )

        # Regenerate with even stricter instructions
        messages.append(
            {
                "role": "system",
                "content": (
                    "WARNING: Your previous response contained information not present in the dataset. "
                    "You MUST only use information explicitly stated in the dataset. "
                    "Do not add ANY details from your general knowledge. "
                    "If you're unsure if something is in the dataset, do not include it. "
                    "Provide a new answer using ONLY information from the dataset."
                ),
            }
        )

        corrected_response = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.0
        )

        answer = corrected_response.choices[0].message.content

    elapsed_time = time.time() - start_time
    if verbose:
        print(
            f"{Symbols.CLOCK}{Colors.GREEN}Total elapsed time: {elapsed_time:.2f} seconds{Colors.ENDC}"
        )

    return answer, source_files


def interactive_mode(verbose=False):
    """Run an interactive session for querying datasets."""
    print(
        f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 20} Vector Database Tool Calling Interactive Mode {'=' * 20}{Colors.ENDC}"
    )
    print(
        f"{Symbols.INFO} Type '{Colors.CYAN}exit{Colors.ENDC}' or '{Colors.CYAN}quit{Colors.ENDC}' to end the session."
    )
    print(
        f"{Symbols.INFO} Type '{Colors.CYAN}verbose{Colors.ENDC}' to toggle verbose mode."
    )
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")

    # Default settings
    model = "gpt-4o-mini"
    verbose_mode = verbose

    # Set up command history
    history_file = os.path.join(os.path.expanduser("~"), ".vector_interactive_history")
    try:
        readline.read_history_file(history_file)
        # Set history file size
        readline.set_history_length(1000)
    except FileNotFoundError:
        # History file doesn't exist yet
        pass

    # Save history on exit
    import atexit

    atexit.register(readline.write_history_file, history_file)

    # Print initial info
    print(f"{Symbols.ROCKET} Using model: {Colors.GREEN}{model}{Colors.ENDC}")
    print(
        f"{Symbols.INFO} Verbose mode: {Colors.GREEN if verbose_mode else Colors.RED}{'on' if verbose_mode else 'off'}{Colors.ENDC}"
    )
    print(f"{Symbols.INFO} Command history enabled (↑/↓ arrows, Ctrl+R for search)")

    # Load vector config
    config = load_vector_config()
    print(
        f"{Symbols.DATABASE} Using Pinecone index: {Colors.CYAN}{config['pinecone_index']}{Colors.ENDC}"
    )

    # Load dataset info
    datasets = load_dataset_info()
    print(
        f"{Symbols.INFO} Available datasets: {', '.join([d['name'] for d in datasets])}"
    )

    while True:
        # Get user input with readline (supports history navigation)
        try:
            query = input(
                f"\n{Colors.BOLD}{Colors.CYAN}Enter your question: {Colors.ENDC}"
            )
        except EOFError:
            # Handle Ctrl+D
            print("\n")
            print(
                f"\n{Symbols.SUCCESS}{Colors.GREEN}Exiting interactive mode.{Colors.ENDC}"
            )
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\n")
            continue

        # Check for special commands
        if query.lower() in ["exit", "quit"]:
            print(
                f"\n{Symbols.SUCCESS}{Colors.GREEN}Exiting interactive mode.{Colors.ENDC}"
            )
            break
        elif query.lower() == "verbose":
            verbose_mode = not verbose_mode
            print(
                f"{Symbols.INFO} Verbose mode: {Colors.GREEN if verbose_mode else Colors.RED}{'on' if verbose_mode else 'off'}{Colors.ENDC}"
            )
            continue
        elif not query.strip():
            continue

        # Process the query
        print(f"\n{Colors.YELLOW}{'-' * 50}{Colors.ENDC}")

        try:
            answer, source_files = process_query(query, model, verbose_mode)

            print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * 50}{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.BLUE}Answer:{Colors.ENDC}")
            print(f"{Colors.CYAN}{answer}{Colors.ENDC}")

            # Display source files
            if source_files:
                print(
                    f"\n{Symbols.FILE} {Colors.BOLD}{Colors.BLUE}Source files:{Colors.ENDC}"
                )
                for file in source_files:
                    print(f"{Symbols.FILE} {Colors.YELLOW}{file}{Colors.ENDC}")
            else:
                print(
                    f"\n{Symbols.WARNING} {Colors.YELLOW}No specific source files identified{Colors.ENDC}"
                )

            print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 50}{Colors.ENDC}")
        except Exception as e:
            print(f"{Symbols.ERROR} {Colors.RED}Error: {str(e)}{Colors.ENDC}")


def run_example_questions(verbose=False):
    """Run a set of example questions."""
    # Load dataset info to get available datasets
    datasets = load_dataset_info()

    # Generate example questions based on available datasets
    questions = []

    for dataset in datasets:
        dataset_name = dataset["name"]
        # Add a generic question for each dataset
        questions.append(f"What information do you have about {dataset_name}?")

        # If we have multiple datasets, add a cross-dataset question
        if len(datasets) > 1 and len(questions) == len(datasets):
            dataset_names = [d["name"] for d in datasets]
            questions.append(f"Compare {' and '.join(dataset_names)}.")

    print(
        f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 25} Running Example Questions {'=' * 25}{Colors.ENDC}"
    )
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 76}{Colors.ENDC}\n")

    # Always use gpt-4o-mini
    model = "gpt-4o-mini"
    print(f"{Symbols.ROCKET} Using model: {Colors.GREEN}{model}{Colors.ENDC}")

    for i, question in enumerate(questions, 1):
        print(
            f"\n{Symbols.QUESTION} {Colors.BOLD}Question {i}/{len(questions)}:{Colors.ENDC}"
        )
        print(f"{Colors.CYAN}'{question}'{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'-' * 50}{Colors.ENDC}")

        answer, source_files = process_query(question, model, verbose)

        print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * 50}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BLUE}Answer:{Colors.ENDC}")
        print(f"{Colors.CYAN}{answer}{Colors.ENDC}")

        # Display source files
        if source_files:
            print(
                f"\n{Symbols.FILE} {Colors.BOLD}{Colors.BLUE}Source files:{Colors.ENDC}"
            )
            for file in source_files:
                print(f"{Symbols.FILE} {Colors.YELLOW}{file}{Colors.ENDC}")
        else:
            print(
                f"\n{Symbols.WARNING} {Colors.YELLOW}No specific source files identified{Colors.ENDC}"
            )

        print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 50}{Colors.ENDC}")

        # Pause between questions to avoid rate limiting
        if i < len(questions):
            print(
                f"\n{Symbols.HOURGLASS} {Colors.YELLOW}Pausing for 2 seconds before next question...{Colors.ENDC}"
            )
            time.sleep(2)

    print(
        f"\n{Symbols.SUCCESS} {Colors.GREEN}All example questions completed!{Colors.ENDC}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Database Tool Calling POC")
    parser.add_argument("--examples", action="store_true", help="Run example questions")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    args = parser.parse_args()

    if args.examples:
        run_example_questions(verbose=args.verbose)
    else:
        interactive_mode(verbose=args.verbose)
