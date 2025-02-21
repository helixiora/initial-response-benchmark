#! /usr/bin/env python3

import os
import time
from datetime import datetime
import json
from typing import Dict, List
from transformers import pipeline
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

# Initialize the classifier
classifier = pipeline(
    "text-classification",
    model="helixiora/distilbert-another-classifier",
    return_all_scores=True,
)

# Label mapping
LABEL_DICT = {
    0: "Clarification",  # Might not need search - asking for clarification
    1: "Factual",  # Likely needs search - asking for facts
    2: "Operational",  # Might not need search - asking how to do something
    3: "Summarization",  # Likely needs search - asking to summarize information
}

# Define which labels typically require search
SEARCH_LABELS = {
    "Factual": True,  # Facts usually need lookup
    "Summarization": True,  # Summarization usually needs source material
    "Clarification": False,  # Clarification usually doesn't need search
    "Operational": False,  # Operational usually doesn't need search
}

# Test questions from the original script
TEST_QUESTIONS = [
    # Questions that likely require search
    {
        "question": "What are the three laws of robotics?",
        "comment": "Factual - Specific knowledge from literature",
        "expects_search": True,
        "expected_class": "Factual",
    },
    {
        "question": "What was the temperature in New York yesterday?",
        "comment": "Factual - Real-time data query",
        "expects_search": True,
        "expected_class": "Factual",
    },
    {
        "question": "What are the latest developments in quantum computing?",
        "comment": "Summarization - Recent developments and current state",
        "expects_search": True,
        "expected_class": "Summarization",
    },
    {
        "question": "Can you help me debug my code?",
        "comment": "Operational - Request for assistance with a task",
        "expects_search": False,
        "expected_class": "Operational",
    },
    {
        "question": "What is 15 * 24?",
        "comment": "Operational - Mathematical operation",
        "expects_search": False,
        "expected_class": "Operational",
    },
    {
        "question": "Convert this temperature: 30°C to Fahrenheit",
        "comment": "Operational - Temperature conversion task",
        "expects_search": False,
        "expected_class": "Operational",
    },
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

        # Determine if we should search based on the predicted label
        should_search = SEARCH_LABELS[predicted_label]

        end_time = time.time()
        time_taken = end_time - start_time

        debug_log(f"Classification completed in {time_taken:.2f}s")
        debug_log(
            f"Predicted label: {predicted_label} (score: {scores[predicted_label]:.4f})"
        )
        debug_log(f"Decision: {'Search' if should_search else 'Skip'}")

        return {
            "success": True,
            "should_search": should_search,
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


def save_results(results: List[Dict], filename: str = "classifier_results.json"):
    """Save results to a JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)


def print_results(results: List[Dict]):
    """Print results in a formatted way with colors."""
    print(f"\n{Fore.WHITE}{Back.BLUE}Classifier Results:{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 100}{Style.RESET_ALL}")

    # Track overall performance
    total = 0
    correct_search = 0
    correct_class = 0
    total_time = 0
    label_counts = {label: 0 for label in LABEL_DICT.values()}
    confusion_matrix = {
        expected: {predicted: 0 for predicted in LABEL_DICT.values()}
        for expected in LABEL_DICT.values()
    }

    for result in results:
        question = result.get("test_case", {}).get("question", "Unknown Question")
        print(f"\n{Fore.GREEN}Question: {Style.BRIGHT}{question}{Style.RESET_ALL}")

        test_case = result.get("test_case", {})
        if test_case:
            expects_search = test_case.get("expects_search", True)
            expected_class = test_case.get("expected_class", "Unknown")
            comment_color = Fore.YELLOW if expects_search else Fore.CYAN
            print(
                f"{comment_color}Expected: {test_case.get('comment', 'No comment provided')}{Style.RESET_ALL}"
            )

        if result["success"]:
            should_search = result["should_search"]
            predicted_label = result["predicted_label"]
            scores = result["label_scores"]
            highest_score = result["highest_score"]
            time_taken = result["time_taken"]

            # Update statistics
            total += 1
            if should_search == expects_search:
                correct_search += 1
            if predicted_label == expected_class:
                correct_class += 1
            total_time += time_taken
            label_counts[predicted_label] += 1
            confusion_matrix[expected_class][predicted_label] += 1

            # Color coding based on correctness and confidence
            search_color = Fore.GREEN if should_search == expects_search else Fore.RED
            class_color = Fore.GREEN if predicted_label == expected_class else Fore.RED
            confidence_color = (
                Fore.GREEN
                if highest_score > 0.5
                else (Fore.YELLOW if highest_score > 0.2 else Fore.RED)
            )

            print(
                f"{search_color}Search Decision: {'Search' if should_search else 'Skip'} (Expected: {'Search' if expects_search else 'Skip'}){Style.RESET_ALL}"
            )
            print(
                f"{class_color}Predicted type: {predicted_label} (Expected: {expected_class}){Style.RESET_ALL}"
            )
            print("Scores:")
            for label, score in scores.items():
                score_color = Fore.GREEN if label == expected_class else Style.RESET_ALL
                print(f"  {score_color}{label}: {score:.4f}{Style.RESET_ALL}")
            print(f"{confidence_color}Confidence: {highest_score:.4f}{Style.RESET_ALL}")
            print(f"Time taken: {time_taken:.2f}s")
        else:
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")

        print(f"{Fore.BLUE}{'-' * 100}{Style.RESET_ALL}")

    # Print summary statistics
    if total > 0:
        search_accuracy = (correct_search / total) * 100
        class_accuracy = (correct_class / total) * 100
        avg_time = total_time / total

        print(f"\n{Fore.WHITE}{Back.BLUE}Performance Summary:{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'=' * 60}{Style.RESET_ALL}")
        print(f"Search Decision Accuracy: {search_accuracy:.1f}%")
        print(f"Class Prediction Accuracy: {class_accuracy:.1f}%")
        print(f"Average time: {avg_time:.3f}s")

        print("\nLabel Distribution:")
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            print(f"{label}: {count} ({percentage:.1f}%)")

        print("\nConfusion Matrix:")
        print("Expected \\ Predicted")
        header = "            " + "".join(
            f"{label:<15}" for label in LABEL_DICT.values()
        )
        print(header)
        print("-" * len(header))
        for expected in LABEL_DICT.values():
            row = f"{expected:<12}"
            for predicted in LABEL_DICT.values():
                count = confusion_matrix[expected][predicted]
                row += f"{count:>14} "
            print(row)

        print(f"\nSearch Decision Mapping:")
        for label, needs_search in SEARCH_LABELS.items():
            print(f"{label}: {'Search' if needs_search else 'Skip'}")


def main():
    results = []
    chat_history = []  # Initialize empty chat history

    for test_case in TEST_QUESTIONS:
        question = test_case["question"]
        print(f"\n{Fore.WHITE}{Back.BLUE}Testing: {question}{Style.RESET_ALL}")

        result = get_classifier_response(question, chat_history)
        result["test_case"] = test_case
        results.append(result)

    print_results(results)
    save_results(results)
    print(
        f"\n{Fore.GREEN}Results have been saved to classifier_results.json{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()
