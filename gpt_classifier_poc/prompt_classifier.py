import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System Instructions
GPT_INSTRUCTIONS = """You are the prompt classification part of a RAG pipeline. 
You will receive a prompt and classify it into only one of the following five categories:

1. **Conversational**: This prompt is purely for conversation and does not require document retrieval. 
   - Examples: "How’s it going?", "Tell me a joke.", "The sky is blue."  
   - These prompts don’t need to make sense, as they aim to get quick responses.

2. **Error**: The input is complete gibberish and lacks any meaning.
   - Examples: "fh sj fsf", "f [QPPF 243%$%6", "sdf ghsd %^4@6".  

3. **Meta Model**: The user is asking about the AI itself.
   - Examples: "What model are you?", "What can you do for me?", "What is your purpose?"

4. **Meta Data**: The user is asking about the data they provided.
   - Examples: "How many PDFs did I give you?", "Summarize the documents I uploaded."

5. **Informational**: Any prompt that does not fit the above categories and requires standard document indexing or 
retrieval.

Respond with only the category name. In the case that the prompt is an error, add a friendly message which states
what you did not understand depending on the type of error alongside the category name in the format "<Category name>. 
<Error message>". In the case the prompt is a  conversational one, add an appropriate response
to the conversation alongside the category name in the format "<Category name>. <Conversation answer>".
"""

def get_response(user_prompt):
    """Sends a prompt to the OpenAI API and returns the classification."""
    messages = [
        {"role": "system", "content": GPT_INSTRUCTIONS},
        {"role": "user", "content": user_prompt}
    ]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return completion.choices[0].message.content.strip()


def main():
    """Runs an interactive input loop for prompt classification."""
    print("Welcome! This is a proof-of-concept for a prompt classifier using OpenAI's GPT-4o-mini.")
    print("Type a message, and the model will classify it into one of 5 categories:")
    print("Conversational, Error, Informational, Meta Data, Meta Model.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter your prompt: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = get_response(user_input)
        print(response)

if __name__ == "__main__":
    main()
