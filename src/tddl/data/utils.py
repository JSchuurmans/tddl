from pathlib import Path

def process_data(path):
    labels = ['cat', 'dog']

    for label in labels:
        path = Path(path)
        target = path + Path(label)
        Path(path).mkdir(parents=True, exist_ok=True)

    