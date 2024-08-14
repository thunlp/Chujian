from pathlib import Path
import random

from data_utils import load_json, extract_entities


MODEL_NAME = "gpt-3.5-turbo"


def parse_preds(pred: str):
    pred = pred.strip()
    lines = pred.split("\n")
    # print("Lines:", lines)
    entities = []
    entity_labels = []
    for line in lines:
        if line.startswith("###"):
            continue
        if line.startswith('-'):
            line = line[1:].strip()
        line = line.replace('：', ':')
        if line:
            word, label = line.split(":")
            word = word.strip()
            label = label.strip()
            entities.append(word)
            entity_labels.append(label)
    return entities, entity_labels


def compute_metrics(label_entities: list[str], labels: list[str], pred_entities: list[str], pred_labels: list[str]):
    '''
    compute recall, precision, f1, accuracy for one example.
    label_entities: list of entities in the label
    labels: list of labels in the label
    pred_entities: list of entities in the prediction
    pred_labels: list of labels in the prediction
    '''
    n = len(label_entities)
    if n == 0:
        return {
            "recall": 0,
            "precision": 0,
            "f1": 0,
            "accuracy": 0,
        }
    n_pred = len(pred_entities)
    n_correct = 0
    for entity, label in zip(label_entities, labels):
        if entity in pred_entities:
            i = pred_entities.index(entity)
            if label == pred_labels[i]:
                n_correct += 1
    recall = n_correct / n
    precision = n_correct / n_pred
    f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0
    accuracy = n_correct / n_pred
    metrics = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
    }
    print(metrics)
    return metrics


def main():
    data_dir = Path("data")
    use_subchars = True
    if use_subchars:
        test_examples = load_json(data_dir / "test_examples_subchars.json")
        # train_examples = load_json(data_dir / "train_examples_subchars.json")
    else:
        test_examples = load_json(data_dir / "test_examples.json")
        # train_examples = load_json(data_dir / "train_examples.json")

    random.seed(0)
    if use_subchars:
        output_dir = Path('result', MODEL_NAME, 'test_subchars')
    else:
        output_dir = Path('result', MODEL_NAME, 'test')
    preds = load_json(output_dir / "preds.json")

    post_metrics = []
    for i, (pred, test_example) in enumerate(zip(preds, test_examples)):
        print(i, "Test example:", test_example)
        label_entities, labels = extract_entities(test_example["input"], test_example["label"])
        label_entities = [''.join(entity) for entity in label_entities]
        print("Label entities:", label_entities)
        print("Labels:", labels)
        pred_entities, pred_labels = parse_preds(pred)
        print("Pred entities:", pred_entities)
        print("Pred labels:", pred_labels)
        metrics = compute_metrics(label_entities, labels, pred_entities, pred_labels)
        post_metrics.append(metrics)

    # Average metrics
    n = len(post_metrics)
    recall = sum([m["recall"] for m in post_metrics]) / n
    precision = sum([m["precision"] for m in post_metrics]) / n
    f1 = sum([m["f1"] for m in post_metrics]) / n
    accuracy = sum([m["accuracy"] for m in post_metrics]) / n
    print("Average metrics:")
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1:", f1)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
