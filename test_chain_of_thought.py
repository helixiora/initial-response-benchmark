#! /usr/bin/env python3

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model to use
MODEL = "gpt-4o-mini"


# Load pricing data
def load_pricing_data() -> Dict:
    """Load pricing data from pricing.json"""
    with open("pricing.json", "r") as f:
        return json.load(f)


# Get token counts for text
def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string for a specific model"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))


# Get the cost per 1K tokens for a model
def get_model_cost(model: str, pricing_data: Dict) -> Dict:
    """Get the cost per 1K tokens for a model"""
    model_key = model

    if model_key in pricing_data:
        model_pricing = pricing_data[model_key]
        # Handle different pricing structures
        if isinstance(model_pricing["input"], str):
            # Extract numerical value from string like "$0.01 / 1K tokens"
            input_cost = float(model_pricing["input"].split("$")[1].split(" ")[0])
            output_cost = float(model_pricing["output"].split("$")[1].split(" ")[0])
        else:
            input_cost = model_pricing["input"]
            output_cost = model_pricing["output"]

        return {"input": input_cost, "output": output_cost}
    else:
        # Default fallback - should log warning here
        return {"input": 0.0, "output": 0.0}


# Calculate cost based on token usage
def calculate_cost(input_tokens: int, output_tokens: int, cost_data: Dict) -> float:
    """Calculate cost based on token usage"""
    input_cost = (input_tokens / 1000) * cost_data["input"]
    output_cost = (output_tokens / 1000) * cost_data["output"]
    return input_cost + output_cost


# Debug log function
def debug_log(msg: str, color: str = Fore.BLUE):
    """Print a debug message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {msg}{Style.RESET_ALL}")


# Chain of thought prompt template
CHAIN_OF_THOUGHT_PROMPT = """
I need to break down a complex question into a chain of thought with individual sub-questions that can be easily answered by a Retrieval Augmented Generation (RAG) system.

The main question is: {question}

Please generate a list of individual sub-questions that form a logical chain of reasoning to answer the main question. 
Each sub-question should:
1. Be specific and focused on a single piece of information
2. Be answerable through knowledge retrieval (facts, definitions, explanations)
3. Follow a logical progression that builds toward answering the main question
4. Be self-contained (can be understood without requiring other sub-questions)
5. Be concise and to the point
6. Be needed to answer the main question, ie. without them you cannot answer the main question

Format your response as a numbered list of questions and generate a flowchart of the sub-questions and their relationships.
"""

# Test questions to analyze
TEST_QUESTIONS = [
    "What is the capital of France?",
    "What is Hristo's date of birth?",
    "Give me a list of the last 10 job applicants for the position of Software Engineer.",
    "Who works in the Marketing department?",
]


def get_chain_of_thought(question: str) -> Dict[str, Any]:
    """Get chain of thought analysis for a question"""
    debug_log(f"Generating chain of thought for: {question}", Fore.GREEN)

    prompt = CHAIN_OF_THOUGHT_PROMPT.format(question=question)

    # Measure time
    start_time = time.time()

    # Get token count for prompt
    prompt_tokens = num_tokens_from_string(prompt, MODEL)

    # Make the API call
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You break down complex questions into specific sub-questions for RAG systems.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Low temperature for consistent results
        )

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Extract response content
        content = response.choices[0].message.content

        # Get token counts and cost
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens

        # Get pricing data
        pricing_data = load_pricing_data()
        cost_data = get_model_cost(MODEL, pricing_data)
        total_cost = calculate_cost(prompt_tokens, completion_tokens, cost_data)

        # Parse sub-questions (simple approach - could be made more robust)
        sub_questions = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 100)):
                question_text = line.split(".", 1)[1].strip()
                sub_questions.append(question_text)

        # Return results
        return {
            "main_question": question,
            "sub_questions": sub_questions,
            "num_sub_questions": len(sub_questions),
            "response_time_seconds": elapsed_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": total_cost,
            "raw_response": content,
        }

    except Exception as e:
        debug_log(f"Error: {str(e)}", Fore.RED)
        return {
            "main_question": question,
            "error": str(e),
            "response_time_seconds": time.time() - start_time,
        }


def save_results(results: List[Dict], filename: str = "chain_of_thought_results.json"):
    """Save results to a JSON file"""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    debug_log(f"Results saved to {filename}", Fore.GREEN)


def print_results_table(results: List[Dict]):
    """Print results in a table format"""
    print("\n" + "=" * 100)
    print(f"{Fore.CYAN}Chain of Thought Analysis Results{Style.RESET_ALL}")
    print("=" * 100)

    headers = ["Question", "Sub-Qs", "Time (s)", "Tokens", "Cost ($)"]
    col_widths = [40, 10, 10, 10, 10]

    # Print headers
    header_row = "".join(f"{h:{w}s}" for h, w in zip(headers, col_widths))
    print(f"{Fore.YELLOW}{header_row}{Style.RESET_ALL}")
    print("-" * 100)

    # Print each result row
    for result in results:
        if "error" in result:
            row = [
                result["main_question"][:37] + "...",
                "ERROR",
                f"{result.get('response_time_seconds', 0):.2f}",
                "N/A",
                "N/A",
            ]
        else:
            row = [
                result["main_question"][:37] + "..."
                if len(result["main_question"]) > 40
                else result["main_question"],
                str(result["num_sub_questions"]),
                f"{result['response_time_seconds']:.2f}",
                str(result["total_tokens"]),
                f"{result['cost_usd']:.6f}",
            ]

        row_str = "".join(f"{col:{w}s}" for col, w in zip(row, col_widths))
        print(row_str)

    # Print summary
    print("-" * 100)
    total_time = sum(r.get("response_time_seconds", 0) for r in results)
    total_cost = sum(r.get("cost_usd", 0) for r in results if "cost_usd" in r)
    total_tokens = sum(r.get("total_tokens", 0) for r in results if "total_tokens" in r)
    avg_sub_questions = (
        sum(r.get("num_sub_questions", 0) for r in results if "num_sub_questions" in r)
        / len([r for r in results if "num_sub_questions" in r])
        if any("num_sub_questions" in r for r in results)
        else 0
    )

    print(f"Total questions: {len(results)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Average sub-questions: {avg_sub_questions:.1f}")
    print("=" * 100)


def display_detailed_results(results: List[Dict]):
    """Display detailed results for each question"""
    for i, result in enumerate(results):
        print("\n" + "=" * 100)
        print(
            f"{Fore.CYAN}Question {i + 1}: {result['main_question']}{Style.RESET_ALL}"
        )
        print("-" * 100)

        if "error" in result:
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            continue

        print(f"Response time: {result['response_time_seconds']:.2f} seconds")
        print(
            f"Token usage: {result['prompt_tokens']} (prompt) + {result['completion_tokens']} (completion) = {result['total_tokens']} total"
        )
        print(f"Cost: ${result['cost_usd']:.6f}")
        print(f"Sub-questions ({result['num_sub_questions']}):")

        for j, question in enumerate(result["sub_questions"]):
            print(f"  {j + 1}. {question}")

    print("\n" + "=" * 100)


def main():
    """Main function"""
    debug_log(f"Starting chain of thought analysis with {MODEL}", Fore.GREEN)

    try:
        # Check for tiktoken
        import tiktoken
    except ImportError:
        debug_log(
            "Error: tiktoken not installed. Please install it with: pip install tiktoken",
            Fore.RED,
        )
        return

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        debug_log(
            "Error: OPENAI_API_KEY not found in environment or .env file", Fore.RED
        )
        return

    # Run the analysis for each test question
    results = []
    for question in TEST_QUESTIONS:
        result = get_chain_of_thought(question)
        results.append(result)
        # Short pause between API calls
        time.sleep(1)

    # Save results
    save_results(results)

    # Print results
    print_results_table(results)

    # Ask if user wants detailed results
    user_input = input(
        f"\n{Fore.YELLOW}Do you want to see detailed results? (y/n): {Style.RESET_ALL}"
    )
    if user_input.lower() == "y":
        display_detailed_results(results)


if __name__ == "__main__":
    main()
