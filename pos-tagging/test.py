from pathlib import Path
import random
from typing import Optional

from openai import OpenAI

from data_utils import create_prompt, load_json, dump_json, verbalize_input

client = OpenAI(api_key="YOUR_API_KEY")
MODEL_NAME = "gpt-4o-mini"


def eval_one(test_input: str, prompt_prefix: str):
    prompt = prompt_prefix + '\n\n' + test_input + '\n### 词:\n'
    messages = [
        {"role": "user", "content": prompt},
    ]
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=200,
    )
    print("Completion:")
    print(completion)
    choices = completion.choices
    print("Choices:")
    print(choices)
    message = choices[0].message
    print("Message:")
    print(message)
    return message.content


def evaluate(test_examples: list, prompt_prefix: str, output_dir: Optional[Path] = None):
    preds = []
    for example in test_examples:
        test_input = verbalize_input(example["input"])
        print("Test input:", test_input)
        response = eval_one(test_input, prompt_prefix)
        print("Response:", response)
        preds.append(response)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            dump_json(preds, output_dir / "preds.json")
    return preds


def main():
    data_dir = Path("data")
    use_subchars = True
    if use_subchars:
        test_examples = load_json(data_dir / "test_examples_subchars.json")
        train_examples = load_json(data_dir / "train_examples_subchars.json")
    else:
        test_examples = load_json(data_dir / "test_examples.json")
        train_examples = load_json(data_dir / "train_examples.json")

    random.seed(0)
    demonstrations = random.sample(train_examples, 10)
    prompt_prefix = create_prompt(demonstrations)
    if use_subchars:
        output_dir = Path('result', MODEL_NAME, 'test_subchars')
    else:
        output_dir = Path('result', MODEL_NAME, 'test')
    evaluate(test_examples, prompt_prefix, output_dir)


if __name__ == "__main__":
    main()
