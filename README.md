# OpenAI Model Response Time Tester

This script tests the response times of different OpenAI models by sending the same question to each model and measuring how long it takes to get a response.

## Prerequisites

- Python 3.11 or newer. The current Torch and Transformers dependencies require Python 3.11.
  On macOS, install a supported version with `brew install python@3.11`.
- pip (comes with Python3)

## Setup

1. Clone this repository

2. Create and activate a virtual environment:
   ```bash
   # Use Python 3.11 or a newer supported interpreter.
   python3.11 -m venv venv
   # On Windows, use: py -3.11 -m venv venv

   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

4. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

5. Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Offline dependency compatibility checks

After installing both `requirements.txt` and `tool_poc/requirements.txt`, run:

```bash
python -m unittest discover -s tests -v
```

These checks use synthetic text and mocked provider clients, block network
connections, and do not download models or make paid API calls. They cover the
vector-tool imports, text chunking, source metadata, and local JSON parsing.
The benchmark scripts at the repository root contact real providers and are
not an offline test suite.

Make sure your virtual environment is activated (you should see `(venv)` at the start of your prompt), then run the script:
```bash
python3 test_response_times.py
```

The script will:
1. Test each model with the same question
2. Measure response times
3. Print results to the console
4. Save detailed results to `response_times.json`

## Customization

To modify the test:
- Edit the `MODELS` list in the script to test different models
- Change the `TEST_QUESTION` variable to test with a different prompt
- Adjust other parameters like temperature in the `get_model_response` function

## Output

The script generates two types of output:
1. Console output with a summary of results
2. A JSON file (`response_times.json`) with detailed results including:
   - Model name
   - Success/failure status
   - Response time
   - Response content (if successful)
   - Error message (if failed)
   - Timestamp

## Deactivating the Virtual Environment

When you're done, you can deactivate the virtual environment:
```bash
deactivate
```

## Troubleshooting

If you see "command not found" errors:
1. Check the interpreter version: `python3 --version` (Python 3.11 or newer is required).
2. If a supported interpreter is missing, on macOS install it with: `brew install python@3.11`
3. Create the virtual environment with `python3.11` or a newer interpreter, then use the activated environment's `python`.
