# OpenAI Model Response Time Tester

This script tests the response times of different OpenAI models by sending the same question to each model and measuring how long it takes to get a response.

## Prerequisites

- Python 3.x (if not installed, on macOS use: `brew install python3`)
- pip (comes with Python3)

## Setup

1. Clone this repository

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python3 -m venv venv

   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
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
1. Make sure Python3 is installed: `which python3`
2. If Python3 is not found, on macOS install it with: `brew install python3`
3. Always use `python3` instead of `python` to ensure you're using Python 3 