# Truths

A simple prompting benchmark for OpenAI language models. It includes the ability to do matching on passed test cases, then rank the test results and output the final accuracy per instruction prompt.

With truths, you can find the best prompt for simple NLP tasks without fine-tuning. There are two learning modes available when testing. You can pick between having a test file using a zero-shot learning model or passing a demonstration file using the few-shot learning model.

## features

- **Zero-shot prompt testing**: test a prompt against a test file to find the best instruction prompt for the task.

- **Few-shot prompt testing**: test a prompt against a test file with a demonstration file to boost accuracy and find the best instruction prompt for the task.

## install

```
pip install -r requirements.txt
```

Dependencies:

- `os` and `sys` for handling file access
- `openai` for access to the various GPT models
- `pyyaml` for parsing yaml files
- `tqdm` for nice looking progress bars
- `argparse` for parsing neat command line arguments
- `prettytable` for simple table printing
- `python-dotenv` for loading env variables from a `.env` file

## example usage

Below is a walkthrough of performing the zero-shot and few-shot learning methods with the provided examples.

```
# zero-shot learning
python truths/truths.py --test "examples/test.yaml"

# few-shot learning
python truths/truths.py --test "examples/test.yaml" --demo "examples/demo.yaml"
```

The output will be represented by a table with the prompts used, accuracy percentages (based on output matching), and the *estimated instruction prompt tokens.

*Note the prompt token count is based on an estimated value of 4.2 characters per token.

## usage
```
truths [-h] [--test file_path] [--demo file_path] [--dry-run] [--debug]
```

## options
- `-h, --help`: Optional flag to show the help message.
- `--test`: String flag for passing the file path for the test file.
- `--demo`: String flag for passing the file path for the demonstration file.
- `--dry-run`: Optional flag to run the program without making API calls.
- `--debug`: Optional flag to run the program in debug mode with verbose output.

## some todos
- get tiktoken working instead of using estimated character/token count value
- create custom ranking algo for rating prompts
- add the winner prompt at the end of the table output computed by using a rating system based on accuracy and prompt token count
- possibly add a feature using an external vector db like Pinecone to test for similarity instead of just output matching
- look into adding the ability to interface with other APIs (Claude, Cohere)

## License
MIT