import os, sys, openai, yaml, tqdm, argparse
from tqdm import tqdm
from collections import defaultdict
from prettytable import PrettyTable, ALL
from dotenv import load_dotenv

class FileNotFoundError(Exception):
    pass

class YAMLError(Exception):
    pass

class Truths:
    def __init__(self, test_file, demo_file, dry_run, debug):
        self.test_file = test_file
        self.dry_run = dry_run
        self.debug = debug
        load_dotenv()
        self.api_key = os.getenv("OPENAI_KEY")
        openai.api_key = self.api_key

        # check if test_file exists
        if not os.path.isfile(test_file):
            raise FileNotFoundError(f"File '{test_file}' not found. Please provide a valid test file path.")

        self.demo_messages = self.load_demo_messages(demo_file) if demo_file else []
        # disable error reporting if debug is false
        if not debug:
            sys.tracebacklimit = 0

    # check if demo_file exists
    def load_demo_messages(self, demo_file):
        demo_messages = []
        try:
            with open(demo_file, 'r') as stream:
                demo_data = yaml.safe_load(stream)['demo']
                for trial_key, trial_value in demo_data.items():
                    if trial_key.startswith('trial-'):
                        for demo_case_num, demo_case_data in trial_value.items():
                            demo_messages.append(
                                {"role": "user", "content": f"{demo_case_data['input']}"})
                            demo_messages.append(
                                {"role": "assistant", "content": f"{demo_case_data['output']}"})
        except (yaml.YAMLError, FileNotFoundError) as exc:
            if self.debug:
                print(exc)
            raise FileNotFoundError(f"File '{demo_file}' not found. Please provide a valid demo file path.")
        return demo_messages

    def process_test(self, test_data):
        results = defaultdict(list)
        for trial_key, trial_value in test_data['test'].items():
            if trial_key.startswith('trial-'):
                prompts = trial_value.get('prompts', [])
                test_cases = trial_value.get('test_cases', {})
                for prompt in prompts:
                    pbar = tqdm(total=len(test_cases), ncols=70)
                    for _,test_case_data in test_cases.items():
                        output, tokens = self.run_prompt(prompt, test_case_data['input'])
                        results[prompt].append((output == test_case_data['output'], tokens))
                        pbar.update()
                    pbar.close()
        return results

    def run_prompt(self, prompt, input):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                # change system message
                {"role": "system", "content": "You must follow the directions given. Do not explain anything."},
                *self.demo_messages,
                {"role": "user", "content": f"{prompt}\n{input}"}
            ]
        )
        return response.choices[0].message['content'].strip(), response['usage']['prompt_tokens']

    def run_test(self):
        try:
            with open(self.test_file, 'r') as stream:
                yaml_data = yaml.safe_load(stream)
                if self.dry_run:
                    print("Dry run finished without errors. Ready for execution.")
                    return
                return self.process_test(yaml_data)
        except (yaml.YAMLError, FileNotFoundError) as exc:
            if self.debug:
                print(exc)
            raise FileNotFoundError(f"File '{self.test_file}' not found. Please provide a valid test file path.")

    def print_results(self, results):
        table = PrettyTable()
        table.field_names = ["Prompt", "Accuracy", "Prompt Tokens"]
        table.align = 'l'
        table._max_width = {"Prompt": 40}
        table.hrules = ALL
        for prompt, result in results.items():
            accuracy = round(sum([res[0] for res in result]) / len(result) * 100)
            prompt_tokens = int(len(prompt) / 4.2)
            table.add_row([prompt, f"{accuracy:}%", prompt_tokens])
        print(table)

def main():
    parser = argparse.ArgumentParser(description='Arguments for truths', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit')
    parser.add_argument('--test', type=str, metavar='file_path', help='Path to the YAML test file')
    parser.add_argument('--demo', type=str, metavar='file_path', help='Path to the YAML demo file')
    parser.add_argument('--dry-run', action='store_true', help='Run the test trials without making API calls')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode for verbose output')

    args = parser.parse_args()

    try:
        truths = Truths(args.test, args.demo, args.dry_run, args.debug)
        results = truths.run_test()
        if results:
            truths.print_results(results)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()