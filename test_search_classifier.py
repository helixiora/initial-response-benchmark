#! /usr/bin/env python3

import os
import time
from datetime import datetime
import json
from typing import Dict, List
from transformers import pipeline
from colorama import init, Fore, Back, Style
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize colorama
init(autoreset=True)

# Initialize the classifier
classifier = pipeline(
    "text-classification",
    model="helixiora/distilbert-another-classifier",
    top_k=None,  # Return scores for all classes
)

# Label mapping
LABEL_DICT = {
    0: "Clarification",  # Asking for explanation or clarification
    1: "Factual",  # Asking for specific facts or information
    2: "Operational",  # Asking how to do something or requesting an action
    3: "Summarization",  # Asking for overview or comparison
}

# Test questions
TEST_QUESTIONS = [
    # Clarification questions
    {
        "question": "What do you mean by 'runtime complexity'?",
        "comment": "Clarification - Asking to explain a term",
        "expected_class": "Clarification",
    },
    {
        "question": "Could you elaborate on what you meant in your last response?",
        "comment": "Clarification - Request to explain previous answer",
        "expected_class": "Clarification",
    },
    {
        "question": "I don't understand the error message. What does it mean?",
        "comment": "Clarification - Understanding an error",
        "expected_class": "Clarification",
    },
    {
        "question": "When you say 'API', which specific one are you referring to?",
        "comment": "Clarification - Disambiguating a term",
        "expected_class": "Clarification",
    },
    {
        "question": "Can you explain what you mean by 'dependency injection'?",
        "comment": "Clarification - Understanding a concept",
        "expected_class": "Clarification",
    },
    # Factual questions
    {
        "question": "What are the three laws of robotics?",
        "comment": "Factual - Specific knowledge from literature",
        "expected_class": "Factual",
    },
    {
        "question": "What was the temperature in New York yesterday?",
        "comment": "Factual - Real-time data query",
        "expected_class": "Factual",
    },
    {
        "question": "When was Python first released?",
        "comment": "Factual - Historical date",
        "expected_class": "Factual",
    },
    {
        "question": "What is the current version of Node.js?",
        "comment": "Factual - Current version information",
        "expected_class": "Factual",
    },
    {
        "question": "Who created the Git version control system?",
        "comment": "Factual - Historical person",
        "expected_class": "Factual",
    },
    # Operational questions
    {
        "question": "Can you help me debug my code?",
        "comment": "Operational - Request for assistance",
        "expected_class": "Operational",
    },
    {
        "question": "What is 15 * 24?",
        "comment": "Operational - Mathematical operation",
        "expected_class": "Operational",
    },
    {
        "question": "Convert this temperature: 30°C to Fahrenheit",
        "comment": "Operational - Temperature conversion",
        "expected_class": "Operational",
    },
    {
        "question": "How do I sort this array in ascending order?",
        "comment": "Operational - Programming task",
        "expected_class": "Operational",
    },
    {
        "question": "Can you help me optimize this function?",
        "comment": "Operational - Code improvement task",
        "expected_class": "Operational",
    },
    # Summarization questions
    {
        "question": "What are the latest developments in quantum computing?",
        "comment": "Summarization - Recent developments overview",
        "expected_class": "Summarization",
    },
    {
        "question": "What are the main differences between REST and GraphQL?",
        "comment": "Summarization - Technology comparison",
        "expected_class": "Summarization",
    },
    {
        "question": "What are the key features of Python 3.11?",
        "comment": "Summarization - Feature overview",
        "expected_class": "Summarization",
    },
    {
        "question": "What are the pros and cons of microservices architecture?",
        "comment": "Summarization - Architecture analysis",
        "expected_class": "Summarization",
    },
    {
        "question": "What are the best practices for React performance optimization?",
        "comment": "Summarization - Best practices overview",
        "expected_class": "Summarization",
    },
]

# GPT-4o prompt template
GPT_PROMPT_TEMPLATE = """Classify the following question into exactly one of these categories:
- Clarification: Questions asking for explanation or clarification of terms, concepts, or previous responses
- Factual: Questions asking for specific facts, data, or information
- Operational: Questions asking how to do something or requesting an action
- Summarization: Questions asking for overview, comparison, or synthesis of information

Question: {question}

Reply with ONLY ONE of these exact category names: Clarification, Factual, Operational, or Summarization.
"""

# Models to test
MODELS = [
    "gpt-4",
    # "gpt-4-turbo-preview",  # This is the current name for gpt-4-0125-preview
    "gpt-4o",
    "gpt-4o-mini",  # Added GPT-4o-mini for comparison
]


def debug_log(msg: str, color: str = Fore.BLUE):
    """Print debug message with timestamp and color"""
    timestamp = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]"
    print(f"{Fore.CYAN}{timestamp} {color}{msg}{Style.RESET_ALL}")


def get_classifier_response(question: str, chat_history: List[Dict] = None) -> Dict:
    """
    Get response from the classifier and measure the time taken.
    """
    chat_history = chat_history or []
    start_time = time.time()

    try:
        # Format the input (similar to minimal template)
        history_str = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in chat_history]
        )
        input_text = f"Query: {question}\nHistory: {history_str}"

        debug_log(f"Calling classifier with input: {input_text}")

        # Get classification
        result = classifier(input_text)[0]

        # Convert scores to dict with actual labels
        scores = {
            LABEL_DICT[int(score["label"].split("_")[-1])]: score["score"]
            for score in result
        }

        # Find the highest scoring label
        predicted_label = max(scores.items(), key=lambda x: x[1])[0]

        end_time = time.time()
        time_taken = end_time - start_time

        debug_log(f"Classification completed in {time_taken:.2f}s")
        debug_log(
            f"Predicted label: {predicted_label} (score: {scores[predicted_label]:.4f})"
        )

        return {
            "success": True,
            "predicted_label": predicted_label,
            "label_scores": scores,
            "highest_score": scores[predicted_label],
            "time_taken": time_taken,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        end_time = time.time()
        debug_log(f"Error: {str(e)}", Fore.RED)
        return {
            "success": False,
            "error": str(e),
            "time_taken": end_time - start_time,
            "timestamp": datetime.now().isoformat(),
        }


def get_gpt_classification(question: str, model: str = "gpt-4o") -> Dict:
    """
    Get classification from GPT model using the prompt template.
    """
    start_time = time.time()
    try:
        # Format the prompt
        prompt = GPT_PROMPT_TEMPLATE.format(question=question)
        debug_log(f"Calling {model} with prompt: {prompt}")

        # Get classification
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Use deterministic output
            max_tokens=10,  # We only need one word
        )

        # Get the predicted label
        predicted_label = response.choices[0].message.content.strip()

        # Validate the response
        if predicted_label not in LABEL_DICT.values():
            raise ValueError(f"Invalid label returned: {predicted_label}")

        end_time = time.time()
        time_taken = end_time - start_time

        debug_log(f"{model} classification completed in {time_taken:.2f}s")
        debug_log(f"Predicted label: {predicted_label}")

        return {
            "success": True,
            "predicted_label": predicted_label,
            "time_taken": time_taken,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "cost": calculate_cost(model, prompt, predicted_label),
        }
    except Exception as e:
        end_time = time.time()
        debug_log(f"{model} Error: {str(e)}", Fore.RED)
        return {
            "success": False,
            "error": str(e),
            "time_taken": end_time - start_time,
            "timestamp": datetime.now().isoformat(),
            "model": model,
        }


def calculate_cost(model: str, prompt: str, response: str) -> float:
    """Calculate the cost of a GPT API call"""
    costs = get_model_cost(model)
    input_tokens = len(prompt.split()) * 1.3  # Rough estimate
    output_tokens = len(response.split()) * 1.3  # Rough estimate
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000


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


def print_combined_summary(
    distilbert_results: List[Dict], gpt_results: Dict[str, List[Dict]]
):
    """Print a combined performance summary comparing all classifiers."""
    print(
        f"\n{Fore.WHITE}{Back.BLUE}Classification Performance Summary{Style.RESET_ALL}"
    )
    print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")

    # Calculate statistics for all models
    stats = {}

    # Add DistilBERT stats
    total = len([r for r in distilbert_results if r["success"]])
    correct = len(
        [
            r
            for r in distilbert_results
            if r["success"] and r["predicted_label"] == r["test_case"]["expected_class"]
        ]
    )
    total_time = sum(r["time_taken"] for r in distilbert_results if r["success"])

    # Calculate label distribution for DistilBERT
    predicted_counts = {label: 0 for label in LABEL_DICT.values()}
    expected_counts = {label: 0 for label in LABEL_DICT.values()}
    for result in distilbert_results:
        if result["success"]:
            predicted_counts[result["predicted_label"]] += 1
            expected_counts[result["test_case"]["expected_class"]] += 1

    stats["DistilBERT"] = {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100) if total > 0 else 0,
        "avg_time": total_time / total if total > 0 else 0,
        "cost": 0,  # DistilBERT has no per-request cost
        "predicted_counts": predicted_counts,
        "expected_counts": expected_counts,
    }

    # Add GPT model stats
    for model, results in gpt_results.items():
        total = len([r for r in results if r["success"]])
        correct = len(
            [
                r
                for r in results
                if r["success"]
                and r["predicted_label"] == r["test_case"]["expected_class"]
            ]
        )
        total_time = sum(r["time_taken"] for r in results if r["success"])
        total_cost = sum(r.get("cost", 0) for r in results if r["success"])

        # Calculate label distribution for this model
        predicted_counts = {label: 0 for label in LABEL_DICT.values()}
        expected_counts = {label: 0 for label in LABEL_DICT.values()}
        for result in results:
            if result["success"]:
                predicted_counts[result["predicted_label"]] += 1
                expected_counts[result["test_case"]["expected_class"]] += 1

        stats[model] = {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "avg_time": total_time / total if total > 0 else 0,
            "cost": total_cost,
            "predicted_counts": predicted_counts,
            "expected_counts": expected_counts,
        }

    # Print model performance
    print("\nModel Performance:")
    print(f"{'-' * 80}")
    for model, model_stats in stats.items():
        accuracy = model_stats["accuracy"]
        avg_time = model_stats["avg_time"]
        cost = model_stats["cost"]

        accuracy_color = (
            Fore.GREEN
            if accuracy >= 80
            else (Fore.YELLOW if accuracy >= 60 else Fore.RED)
        )

        print(f"{model}:")
        print(f"  Accuracy: {accuracy_color}{accuracy:>6.1f}%{Style.RESET_ALL}")
        print(f"  Avg Response Time: {avg_time:.3f}s")
        if cost > 0:
            print(f"  Total Cost: ${cost:.4f}")

    # Calculate and display speed comparison
    base_time = stats["DistilBERT"]["avg_time"]
    print("\nSpeed Comparison (vs DistilBERT):")
    for model in stats:
        if model != "DistilBERT":
            ratio = stats[model]["avg_time"] / base_time
            print(f"  {model}: {ratio:.1f}x slower")

    # Print label distribution
    print(f"\n{Fore.WHITE}Label Distribution Analysis{Style.RESET_ALL}")
    print(f"{'-' * 120}")

    # Create header with all models
    header = f"{'Label':<15} {'Expected':<10}"
    for model in stats:
        header += f"{model:<20}"
    print(header)
    print("-" * 120)

    # Print distribution for each label
    for label in LABEL_DICT.values():
        row = f"{label:<15}"
        expected = stats["DistilBERT"]["expected_counts"][label]  # Same for all models
        row += f"{expected:>3}        "

        for model in stats:
            predicted = stats[model]["predicted_counts"][label]
            diff = predicted - expected
            # Color coding for differences
            diff_color = (
                Fore.GREEN
                if diff == 0
                else (Fore.YELLOW if abs(diff) == 1 else Fore.RED)
            )
            row += f"{predicted:>3} {diff_color}({diff:>+3}){Style.RESET_ALL}         "

        print(row)

    print("\nKey:")
    print(f"{Fore.GREEN}Green{Style.RESET_ALL}: Perfect match")
    print(f"{Fore.YELLOW}Yellow{Style.RESET_ALL}: Small difference (±1)")
    print(f"{Fore.RED}Red{Style.RESET_ALL}: Larger difference")


def main():
    distilbert_results = []
    gpt_results = {model: [] for model in MODELS}
    chat_history = []  # Initialize empty chat history

    print(f"\n{Fore.WHITE}{Back.BLUE}Running Classification Tests{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 100}{Style.RESET_ALL}")

    for test_case in TEST_QUESTIONS:
        question = test_case["question"]
        print(f"\n{Fore.WHITE}{Back.BLUE}Testing: {question}{Style.RESET_ALL}")

        # Get DistilBERT classification
        print(f"\n{Fore.YELLOW}DistilBERT Classification:{Style.RESET_ALL}")
        distilbert_result = get_classifier_response(question, chat_history)
        distilbert_result["test_case"] = test_case
        distilbert_results.append(distilbert_result)

        # Get GPT classifications
        for model in MODELS:
            print(f"\n{Fore.YELLOW}{model} Classification:{Style.RESET_ALL}")
            gpt_result = get_gpt_classification(question, model)
            gpt_result["test_case"] = test_case
            gpt_results[model].append(gpt_result)

        # Print individual results
        print(f"\n{Fore.CYAN}Results Comparison:{Style.RESET_ALL}")
        print(f"{'Expected:':<12} {test_case['expected_class']}")

        # Print DistilBERT result
        if distilbert_result["success"]:
            db_correct = (
                distilbert_result["predicted_label"] == test_case["expected_class"]
            )
            print(
                f"{'DistilBERT:':<12} {Fore.GREEN if db_correct else Fore.RED}"
                f"{distilbert_result['predicted_label']}{Style.RESET_ALL}"
            )

        # Print GPT results
        for model in MODELS:
            result = gpt_results[model][-1]  # Get the latest result for this model
            if result["success"]:
                correct = result["predicted_label"] == test_case["expected_class"]
                print(
                    f"{model + ':':<12} {Fore.GREEN if correct else Fore.RED}"
                    f"{result['predicted_label']}{Style.RESET_ALL}"
                )

    # Print combined summary
    print_combined_summary(distilbert_results, gpt_results)

    # Save results
    all_results = {
        "distilbert": distilbert_results,
        "gpt": gpt_results,
        "timestamp": datetime.now().isoformat(),
    }
    with open("classifier_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(
        f"\n{Fore.GREEN}Results have been saved to classifier_results.json{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()
