# Dataset Tool Calling POC

This proof of concept demonstrates how to use text datasets as tools with OpenAI's API. Each dataset contains a set of text files about a specific subject, and the application allows you to ask questions that get answered using these datasets as tools.

## Datasets

The application includes three example datasets:

1. **Pokemon** - Information about Pokemon, including starters, legendary Pokemon, and type effectiveness.
2. **Space** - Information about celestial objects and phenomena in our universe.
3. **History** - Information about significant historical periods and their impact on human civilization.

## Setup

1. Clone this repository
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   cp .env.example .env
   # Edit .env to add your OpenAI API key
   ```

## Usage

### Interactive Mode

Start an interactive session for querying datasets:

```bash
python interactive.py
```

In interactive mode, you can:
- Ask questions about any dataset
- Type `verbose` to toggle verbose mode (shows more details about the process)
- Type `model <name>` to change the OpenAI model (e.g., `model gpt-3.5-turbo`)
- Type `exit` or `quit` to end the session

### Run Example Questions

Run a set of predefined example questions:

```bash
python interactive.py --examples
```

## How It Works

1. The application loads all datasets from the `datasets` directory.
2. Each dataset becomes a tool that the LLM can use to answer questions.
3. When you ask a question, the LLM determines which dataset(s) to use.
4. The appropriate dataset tool is called to retrieve information.
5. The LLM uses this information to formulate a response.
6. A validation step ensures the response only contains information from the dataset.

## Adding New Datasets

To add a new dataset:

1. Create a new directory under `datasets/`
2. Add text files with information
3. Create a `metadata.json` file with the following structure:
   ```json
   {
     "name": "Dataset Name",
     "description": "Description of the dataset",
     "files": [
       {
         "filename": "file1.txt",
         "description": "Description of file1"
       },
       {
         "filename": "file2.txt",
         "description": "Description of file2"
       }
     ]
   }
   ```

The application will automatically detect and load the new dataset as a tool. 