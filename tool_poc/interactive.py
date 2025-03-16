#!/usr/bin/env python3
"""
Interactive script to demonstrate dataset tool calling.
This directly uses the OpenAI API to implement tool calling.
"""

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please create a .env file based on .env.example and add your API key.")
    exit(1)

# Initialize OpenAI client
client = OpenAI()


def load_dataset_metadata(datasets_dir: str) -> List[Dict[str, Any]]:
    """Load metadata for all datasets."""
    datasets = []

    # Get all subdirectories in the datasets directory
    dataset_dirs = [
        d
        for d in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, d))
    ]

    for dataset_dir in dataset_dirs:
        dataset_path = os.path.join(datasets_dir, dataset_dir)
        metadata_path = os.path.join(dataset_path, "metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Add path to metadata
            metadata["path"] = dataset_path
            datasets.append(metadata)

    return datasets


def get_dataset_content(dataset_path: str, files: List[Dict[str, str]]) -> str:
    """Get the content of all files in a dataset."""
    results = []

    for file_info in files:
        filename = file_info["filename"]
        file_path = os.path.join(dataset_path, filename)

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                results.append(f"\nContent from {filename}:\n{content}")

    return "\n".join(results)


def create_tools(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create tool definitions for OpenAI API."""
    tools = []

    for dataset in datasets:
        # Create more explicit tool descriptions
        if dataset["name"] == "Pokemon":
            description = "Use this tool for any questions about Pokemon, including starter Pokemon, legendary Pokemon, Pokemon types, and type effectiveness."
        elif dataset["name"] == "Space":
            description = "Use this tool for any questions about space, astronomy, planets, galaxies, black holes, and other celestial objects."
        elif dataset["name"] == "History":
            description = "Use this tool for any questions about historical periods, including Ancient Rome, the Renaissance, and the Industrial Revolution."
        else:
            description = (
                f"Use this tool to answer questions about {dataset['description']}"
            )

        tool = {
            "type": "function",
            "function": {
                "name": f"get_{dataset['name'].lower()}_info",
                "description": description,
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


def process_query(query: str, model: str = "gpt-4o", verbose: bool = False):
    """Process a user query using dataset tools."""
    # Load datasets
    datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
    datasets = load_dataset_metadata(datasets_dir)

    # Create tools
    tools = create_tools(datasets)

    if verbose:
        # Print available datasets
        print(f"Loaded {len(datasets)} dataset tools:")
        for dataset in datasets:
            print(f"- {dataset['name']}: {dataset['description']}")

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
        print("Asking model which tool to use...")

    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto"
    )

    tool_calls = response.choices[0].message.tool_calls

    if not tool_calls:
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"No tool calls made. Elapsed time: {elapsed_time:.2f} seconds")
        return "I couldn't determine which dataset to use for your question. Please try asking about Pokemon, Space, or History."

    # Step 2: Execute the tool call
    tool_call = tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    # Find the dataset
    dataset_name = function_name.split("_")[1].capitalize()
    dataset = next(
        (d for d in datasets if d["name"].lower() == dataset_name.lower()), None
    )

    if verbose:
        print(f"Using dataset: {dataset_name}")

    if not dataset:
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Dataset not found. Elapsed time: {elapsed_time:.2f} seconds")
        return f"Dataset {dataset_name} not found."

    # Get dataset content
    if verbose:
        print(f"Retrieving content from {dataset_name} dataset...")

    dataset_content = get_dataset_content(dataset["path"], dataset["files"])

    # Step 3: Get the final answer with strict instructions
    if verbose:
        print("Generating final answer...")

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

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,  # Use lower temperature for more deterministic responses
    )

    # Step 4: Validate the response against the dataset content
    answer = final_response.choices[0].message.content

    if verbose:
        print("Validating answer against dataset content...")

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

    validation_response = client.chat.completions.create(
        model=model, messages=validation_messages, temperature=0.0
    )

    validation_result = validation_response.choices[0].message.content

    # If the validation fails, regenerate the answer with stricter constraints
    if not validation_result.startswith("VALID"):
        if verbose:
            print(f"Validation failed: {validation_result}")
            print("Regenerating answer with stricter constraints...")

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

        corrected_response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0
        )

        answer = corrected_response.choices[0].message.content

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")

    return answer


def interactive_mode():
    """Run an interactive session for querying datasets."""
    print("Dataset Tool Calling Interactive Mode")
    print("====================================")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'model <name>' to change the OpenAI model.")
    print("Type 'verbose' to toggle verbose mode.")

    # Default settings
    model = "gpt-4o"
    verbose = False

    # Print initial info
    print(f"Using model: {model}")
    print(f"Verbose mode: {'on' if verbose else 'off'}")

    while True:
        # Get user input
        query = input("\nEnter your question: ")

        # Check for special commands
        if query.lower() in ["exit", "quit"]:
            print("Exiting interactive mode.")
            break
        elif query.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'on' if verbose else 'off'}")
            continue
        elif query.lower().startswith("model "):
            model = query.split(" ", 1)[1].strip()
            print(f"Switched to model: {model}")
            continue
        elif not query.strip():
            continue

        # Process the query
        print("-" * 50)

        try:
            answer = process_query(query, model, verbose)

            print("\n" + "=" * 50)
            print("Answer:")
            print(answer)
            print("=" * 50)
        except Exception as e:
            print(f"Error: {str(e)}")


def run_example_questions():
    """Run a set of example questions."""
    questions = [
        # Pokemon questions
        "What are the starter Pokemon in Generation I?",
        "Tell me about the legendary birds in Pokemon.",
        "What types are effective against Ghost Pokemon?",
        # Space questions
        "What are the planets in our solar system?",
        "Explain the different types of galaxies.",
        "What are the main types of black holes?",
        # History questions
        "What were the key periods of Ancient Rome?",
        "What were the main features of the Renaissance?",
        "How did the Industrial Revolution change society?",
        # Cross-dataset question
        "Compare the structure of our solar system with the structure of Ancient Rome.",
    ]

    print("Running Example Questions")
    print("========================")

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}:")
        print(f"'{question}'")
        print("-" * 50)

        answer = process_query(question, verbose=True)

        print("\n" + "=" * 50)
        print("Answer:")
        print(answer)
        print("=" * 50)

        # Pause between questions to avoid rate limiting
        if i < len(questions):
            print("\nPausing for 2 seconds before next question...")
            time.sleep(2)

    print("\nAll example questions completed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        run_example_questions()
    else:
        interactive_mode()
