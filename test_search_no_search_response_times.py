#! /usr/bin/env python3

import os
import time
from datetime import datetime
import json
from typing import Dict, List
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Models to test
MODELS = [
    "gpt-4",
    # "gpt-4-turbo-preview",  # This is the current name for gpt-4-0125-preview
    "gpt-4o",
    # "gpt-4o-mini",
]

# Search determination prompt templates
PROMPT_TEMPLATES = [
    # Alternative 0: Minimal format
    {
        "name": "Minimal",
        "template": """Query: {final_query}
History: {chat_history}

Does this need external information to answer?
Reply EXACTLY "Yes Search" or "Skip Search".
Skip Search only if basic knowledge/math/coding is sufficient.""",
    },
    # Alternative 1: Direct question format
    {
        "name": "Direct Question",
        "template": """Based on the conversation below, should we search for external information to answer the latest query?

Reply ONLY with "Yes Search" or "Skip Search".
Default to "Yes Search" if unsure.

Choose "Skip Search" only if:
1. The chat history contains ALL needed information to give a complete answer, or
2. The query can be handled without additional facts or data

Previous conversation:
===
{chat_history}
===

New query: {final_query}

Remember: If in doubt, say "Yes Search".""",
    },
    # Alternative 2: Checklist format
    {
        "name": "Checklist",
        "template": """Search Requirement Checklist

Context:
- Chat history is below
- New query: {final_query}
- Must reply EXACTLY "Yes Search" or "Skip Search"

Chat History:
===
{chat_history}
===

Answer "Skip Search" if ALL are true:
✓ We can fully answer without external data
✓ Chat history has all needed information (if any needed)
✓ No current/real-time information required
✓ No additional context would improve the answer

Otherwise, answer "Yes Search"

Your response (Yes Search/Skip Search):""",
    },
    # Alternative 3: Decision tree format
    {
        "name": "Decision Tree",
        "template": """Search Decision Tree
Query: {final_query}

History:
===
{chat_history}
===

1. Can the query be answered with:
   - Basic knowledge
   - Simple calculations
   - Code examples
   - Generic advice
   If YES → consider "Skip Search"
   If NO → respond "Yes Search"

2. If considering "Skip Search", verify:
   - No real-world data needed
   - No time-sensitive info needed
   - No specific facts required
   - Chat history contains all context
   
If ANY verification fails → "Yes Search"
If ALL pass → "Skip Search"

Your decision (respond EXACTLY with "Yes Search" or "Skip Search"):""",
    },
]

# Test questions
TEST_QUESTIONS = [
    # Questions that likely require search
    {
        "question": "What are the three laws of robotics?",
        "comment": "Requires search - Specific factual knowledge from Asimov's works",
        "expects_search": True,
    },
    {
        "question": "What was the temperature in New York yesterday?",
        "comment": "Requires search - Real-time/historical data needed",
        "expects_search": True,
    },
    {
        "question": "What are the latest developments in quantum computing?",
        "comment": "Requires search - Current events and recent developments",
        "expects_search": True,
    },
    # {
    #     "question": "Who won the most recent Super Bowl and what was the score?",
    #     "comment": "Requires search - Recent sports event information",
    #     "expects_search": True,
    # },
    # {
    #     "question": "What is the current version of Python and its new features?",
    #     "comment": "Requires search - Version information changes over time",
    #     "expects_search": True,
    # },
    # Questions that likely don't require search
    {
        "question": "Can you help me debug my code?",
        "comment": "Skip search - Generic request for assistance",
        "expects_search": False,
    },
    {
        "question": "What is 15 * 24?",
        "comment": "Skip search - Simple mathematical calculation",
        "expects_search": False,
    },
    {
        "question": "Convert this temperature: 30°C to Fahrenheit",
        "comment": "Skip search - Simple conversion using known formula",
        "expects_search": False,
    },
    # {
    #     "question": "Write a function to reverse a string in Python",
    #     "comment": "Skip search - Basic programming task",
    #     "expects_search": False,
    # },
    # {
    #     "question": "Explain what a try-except block does",
    #     "comment": "Skip search - Basic programming concept explanation",
    #     "expects_search": False,
    # },
]


def debug_log(msg: str, color: str = Fore.BLUE):
    """Print debug message with timestamp and color"""
    timestamp = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]"
    print(f"{Fore.CYAN}{timestamp} {color}{msg}{Style.RESET_ALL}")


def debug_log_prompt(prompt: str):
    """Print a formatted prompt with timestamp"""
    timestamp = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]"
    print(f"{Fore.CYAN}{timestamp} {Fore.YELLOW}Sending prompt:{Style.RESET_ALL}")
    print(f"{Fore.BLACK}{Back.WHITE}{'-' * 80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{prompt}{Style.RESET_ALL}")
    print(f"{Fore.BLACK}{Back.WHITE}{'-' * 80}{Style.RESET_ALL}")


def should_search(chat_history: List[Dict], query: str) -> tuple:
    """
    Determine if we should search for additional information based on chat history and query.
    """
    try:
        # Format chat history into string
        history_str = "\n".join(
            [
                f"User: {msg['role'] == 'user'}\n{msg['content']}\n"
                for msg in chat_history
            ]
        )

        # Try each prompt template
        results = []
        template_times = {}
        template_decisions = {}

        for template in PROMPT_TEMPLATES:
            template_start = time.time()
            # Format the prompt
            prompt = template["template"].format(
                chat_history=history_str, final_query=query
            )

            # Get determination from OpenAI
            debug_log(
                f"Calling GPT-4-turbo for search determination using {template['name']} template..."
            )
            debug_log_prompt(prompt)

            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            template_end = time.time()
            template_times[template["name"]] = template_end - template_start

            debug_log("Received search determination response")

            determination = response.choices[0].message.content.strip()
            template_decisions[template["name"]] = determination
            debug_log(f"Search determination ({template['name']}): {determination}")
            results.append(determination == "Yes Search")

        # Take majority vote
        should_search = sum(results) > len(results) / 2
        debug_log(
            f"Final determination (majority vote): {'Yes Search' if should_search else 'Skip Search'}"
        )
        return should_search, template_times, template_decisions

    except Exception as e:
        print(f"Error in search determination: {e}")
        return True, {}, {}  # Default to Yes if there's an error


def get_model_cost(model: str) -> Dict:
    """Get the cost per 1K tokens for a model"""
    # Default to GPT-4 Turbo pricing if model not found
    costs = {
        "gpt-4": {
            "input": 0.03,  # $0.03 per 1K tokens
            "output": 0.06,  # $0.06 per 1K tokens
        },
        "gpt-4-turbo-preview": {
            "input": 0.01,  # $0.01 per 1K tokens
            "output": 0.03,  # $0.03 per 1K tokens
        },
        "gpt-4o": {
            "input": 0.0025,  # $0.0025 per 1K tokens
            "output": 0.01,  # $0.01 per 1K tokens
        },
        "gpt-4o-mini": {
            "input": 0.00015,  # $0.00015 per 1K tokens
            "output": 0.0006,  # $0.0006 per 1K tokens
        },
    }
    return costs.get(model, costs["gpt-4-turbo-preview"])


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost for a model run"""
    costs = get_model_cost(model)
    input_cost = (input_tokens / 1000) * costs["input"]
    output_cost = (output_tokens / 1000) * costs["output"]
    return input_cost + output_cost


def get_model_response(
    model: str, question: str, chat_history: List[Dict] = None
) -> Dict:
    """
    Get response from a specific model and measure the time taken.
    """
    chat_history = chat_history or []
    start_time = time.time()
    try:
        # First determine if we should search
        debug_log(f"Starting search determination for model {model}")
        should_search_result, template_times, template_decisions = should_search(
            chat_history, question
        )
        search_determination_time = time.time() - start_time
        debug_log(f"Search determination took {search_determination_time:.2f}s")

        # Get the actual response
        debug_log(f"Calling {model} for main response...")
        debug_log_prompt(question)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
        )
        debug_log(f"Received response from {model}")
        end_time = time.time()

        # Calculate costs
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_cost = calculate_cost(model, prompt_tokens, completion_tokens)

        return {
            "model": model,
            "success": True,
            "response": response.choices[0].message.content,
            "time_taken": end_time - start_time,
            "search_determination_time": search_determination_time,
            "template_times": template_times,
            "template_decisions": template_decisions,
            "should_search": should_search_result,
            "timestamp": datetime.now().isoformat(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_cost": total_cost,
        }
    except Exception as e:
        end_time = time.time()
        debug_log(f"Error with {model}: {str(e)}")
        return {
            "model": model,
            "success": False,
            "error": str(e),
            "time_taken": end_time - start_time,
            "timestamp": datetime.now().isoformat(),
        }


def save_results(results: List[Dict], filename: str = "response_times.json"):
    """
    Save results to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)


def print_results(results: List[Dict]):
    """
    Print results in a formatted way with colors.
    """
    print(f"\n{Fore.WHITE}{Back.BLUE}Response Time Results:{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")

    # Group results by question
    questions = {}
    for result in results:
        question = result.get("test_case", {}).get("question", "Unknown Question")
        if question not in questions:
            questions[question] = []
        questions[question].append(result)

    # Print results grouped by question
    for question, question_results in questions.items():
        print(f"\n{Fore.GREEN}Question: {Style.BRIGHT}{question}{Style.RESET_ALL}")
        test_case = question_results[0].get("test_case", {})
        if test_case:
            expects_search = test_case.get("expects_search", True)
            comment_color = Fore.YELLOW if expects_search else Fore.CYAN
            print(
                f"{comment_color}Expected behavior: {test_case.get('comment', 'No comment provided')}{Style.RESET_ALL}"
            )
        print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")

        for result in question_results:
            print(
                f"\n{Fore.MAGENTA}Model: {Style.BRIGHT}{result['model']}{Style.RESET_ALL}"
            )
            success = result["success"]
            success_color = Fore.GREEN if success else Fore.RED
            print(f"{success_color}Success: {success}{Style.RESET_ALL}")
            print(
                f"{Fore.BLUE}Time taken: {Style.BRIGHT}{result['time_taken']:.2f} seconds{Style.RESET_ALL}"
            )

            if "search_determination_time" in result:
                print(
                    f"{Fore.BLUE}Search determination time: {Style.BRIGHT}{result['search_determination_time']:.2f} seconds{Style.RESET_ALL}"
                )
                should_search = result.get("should_search", False)
                expected_search = test_case.get("expects_search", True)
                search_color = (
                    Fore.GREEN if should_search == expected_search else Fore.RED
                )
                print(
                    f"{search_color}Should search: {should_search} (Expected: {expected_search}){Style.RESET_ALL}"
                )

            if success:
                print(
                    f"{Fore.BLUE}Response length: {Style.BRIGHT}{len(result['response'])} characters{Style.RESET_ALL}"
                )
            else:
                print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")


def print_summary_table(results: List[Dict]):
    """
    Print a summary table comparing model and template combinations for speed, accuracy, and cost.
    """
    print(
        f"\n{Fore.WHITE}{Back.BLUE}Model & Template Performance Analysis{Style.RESET_ALL}"
    )
    print(f"{Fore.BLUE}{'=' * 160}{Style.RESET_ALL}")

    # Track performance stats by model and template
    performance_stats = {}
    for model in MODELS:
        performance_stats[model] = {
            template["name"]: {
                "correct": 0,
                "total": 0,
                "total_time": 0,
                "avg_time": 0,
                "fastest_time": float("inf"),
                "slowest_time": 0,
                "total_cost": 0,
                "avg_cost": 0,
            }
            for template in PROMPT_TEMPLATES
        }

    # Collect statistics
    for result in results:
        if not result.get("success", False):
            continue

        model = result["model"]
        test_case = result.get("test_case", {})
        expected_search = test_case.get("expects_search", True)
        template_times = result.get("template_times", {})
        template_decisions = result.get("template_decisions", {})
        total_cost = result.get("total_cost", 0)

        for template in PROMPT_TEMPLATES:
            template_name = template["name"]
            decision = template_decisions.get(template_name, "N/A")
            time_taken = template_times.get(template_name, 0)

            stats = performance_stats[model][template_name]
            stats["total"] += 1
            if (decision == "Yes Search") == expected_search:
                stats["correct"] += 1
            stats["total_time"] += time_taken
            stats["avg_time"] = stats["total_time"] / stats["total"]
            stats["fastest_time"] = min(stats["fastest_time"], time_taken)
            stats["slowest_time"] = max(stats["slowest_time"], time_taken)
            stats["total_cost"] += total_cost / len(
                PROMPT_TEMPLATES
            )  # Split cost among templates
            stats["avg_cost"] = stats["total_cost"] / stats["total"]

    # Print header
    header = f"{Fore.WHITE}{Style.BRIGHT}"
    header += f"{'Model':<15} {'Template':<20} {'Accuracy':<10} {'Avg Time':<10} {'Best Time':<10} {'Cost/Query':<10} {'Score':<10}"
    print(header + Style.RESET_ALL)
    print(f"{Fore.BLUE}{'-' * 160}{Style.RESET_ALL}")

    # Track best performers
    best_accuracy = {"score": 0, "combinations": []}
    best_speed = {"time": float("inf"), "combinations": []}
    best_cost = {"cost": float("inf"), "combinations": []}
    best_overall = {"score": 0, "combinations": []}

    # Print each model-template combination
    for model in MODELS:
        print(f"{Fore.WHITE}{Back.BLUE}{model:<160}{Style.RESET_ALL}")
        for template_name, stats in performance_stats[model].items():
            accuracy = (
                (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            )
            avg_time = stats["avg_time"]
            avg_cost = stats["avg_cost"]

            # Calculate combined score (60% accuracy, 20% speed, 20% cost)
            time_score = max(0, 1 - (avg_time / 5))  # Normalize time (5s max)
            cost_score = max(0, 1 - (avg_cost / 0.01))  # Normalize cost ($0.01 max)
            combined_score = (accuracy * 0.6) + (time_score * 20) + (cost_score * 20)

            # Update best performers
            combination = {"model": model, "template": template_name}

            if accuracy > best_accuracy["score"]:
                best_accuracy["score"] = accuracy
                best_accuracy["combinations"] = [combination]
            elif accuracy == best_accuracy["score"]:
                best_accuracy["combinations"].append(combination)

            if avg_time < best_speed["time"]:
                best_speed["time"] = avg_time
                best_speed["combinations"] = [combination]
            elif avg_time == best_speed["time"]:
                best_speed["combinations"].append(combination)

            if avg_cost < best_cost["cost"]:
                best_cost["cost"] = avg_cost
                best_cost["combinations"] = [combination]
            elif avg_cost == best_cost["cost"]:
                best_cost["combinations"].append(combination)

            if combined_score > best_overall["score"]:
                best_overall["score"] = combined_score
                best_overall["combinations"] = [combination]
            elif combined_score == best_overall["score"]:
                best_overall["combinations"].append(combination)

            # Color coding
            accuracy_color = (
                Fore.GREEN
                if accuracy >= 80
                else (Fore.YELLOW if accuracy >= 60 else Fore.RED)
            )
            time_color = (
                Fore.GREEN
                if avg_time < 2
                else (Fore.YELLOW if avg_time < 3 else Fore.RED)
            )
            cost_color = (
                Fore.GREEN
                if avg_cost < 0.001
                else (Fore.YELLOW if avg_cost < 0.005 else Fore.RED)
            )
            score_color = (
                Fore.GREEN
                if combined_score >= 80
                else (Fore.YELLOW if combined_score >= 60 else Fore.RED)
            )

            row = f"{' ' * 15}"  # Model already shown in header
            row += f"{Fore.CYAN}{template_name:<20}{Style.RESET_ALL} "
            row += f"{accuracy_color}{accuracy:>6.1f}%{Style.RESET_ALL}   "
            row += f"{time_color}{avg_time:>7.2f}s{Style.RESET_ALL}  "
            row += f"{Fore.BLUE}{stats['fastest_time']:>7.2f}s{Style.RESET_ALL}  "
            row += f"{cost_color}${avg_cost:>8.4f}{Style.RESET_ALL}  "
            row += f"{score_color}{combined_score:>7.1f}{Style.RESET_ALL}"
            print(row)
        print(f"{Fore.BLUE}{'-' * 160}{Style.RESET_ALL}")

    # Print best performers summary
    print(f"\n{Fore.WHITE}{Back.BLUE}Best Performers{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")

    print(
        f"{Fore.GREEN}Most Accurate ({best_accuracy['score']:.1f}%):{Style.RESET_ALL}"
    )
    for combo in best_accuracy["combinations"]:
        print(
            f"{Fore.GREEN}  • {Style.BRIGHT}{combo['model']} with {combo['template']}{Style.RESET_ALL}"
        )

    print(f"\n{Fore.YELLOW}Fastest ({best_speed['time']:.2f}s):{Style.RESET_ALL}")
    for combo in best_speed["combinations"]:
        print(
            f"{Fore.YELLOW}  • {Style.BRIGHT}{combo['model']} with {combo['template']}{Style.RESET_ALL}"
        )

    print(
        f"\n{Fore.MAGENTA}Most Cost-Effective (${best_cost['cost']:.4f}/query):{Style.RESET_ALL}"
    )
    for combo in best_cost["combinations"]:
        print(
            f"{Fore.MAGENTA}  • {Style.BRIGHT}{combo['model']} with {combo['template']}{Style.RESET_ALL}"
        )

    print(
        f"\n{Fore.CYAN}Best Overall (Score: {best_overall['score']:.1f}):{Style.RESET_ALL}"
    )
    for combo in best_overall["combinations"]:
        print(
            f"{Fore.CYAN}  • {Style.BRIGHT}{combo['model']} with {combo['template']}{Style.RESET_ALL}"
        )

    print(f"\nNote: Overall score weights: Accuracy 60%, Speed 20%, Cost 20%")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Fore.RED}Error: OPENAI_API_KEY not found in environment variables")
        print(f"Please create a .env file with your OpenAI API key{Style.RESET_ALL}")
        return

    results = []
    chat_history = []  # Initialize empty chat history

    for test_case in TEST_QUESTIONS:
        question = test_case["question"]
        print(f"\n{Fore.WHITE}{Back.BLUE}{'=' * 80}{Style.RESET_ALL}")
        print(
            f"{Fore.GREEN}Testing question: '{Style.BRIGHT}{question}{Style.RESET_ALL}'"
        )
        print(
            f"{Fore.YELLOW}Expected behavior: {test_case['comment']}{Style.RESET_ALL}"
        )
        print(f"{Fore.WHITE}{Back.BLUE}{'=' * 80}{Style.RESET_ALL}")

        for model in MODELS:
            print(f"\n{Fore.MAGENTA}Testing {Style.BRIGHT}{model}...{Style.RESET_ALL}")
            result = get_model_response(model, question, chat_history)
            result["test_case"] = test_case  # Include the test case info in results
            results.append(result)
            # # Add to chat history if successful
            # if result["success"]:
            #     chat_history.append({"role": "user", "content": question})
            #     chat_history.append(
            #         {"role": "assistant", "content": result["response"]}
            #     )

    print_results(results)
    print_summary_table(results)  # Add summary table
    save_results(results)
    print(
        f"\n{Fore.GREEN}Results have been saved to response_times.json{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()
