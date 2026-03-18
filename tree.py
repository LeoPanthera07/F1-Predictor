import os

def tree(path, prefix=''):
    entries = sorted(os.listdir(path))
    skip = {'__pycache__', 'node_modules', '.git', 'dist'}
    entries = [e for e in entries if e not in skip]
    for i, entry in enumerate(entries):
        full = os.path.join(path, entry)
        connector = '└── ' if i == len(entries)-1 else '├── '
        if os.path.isfile(full):
            size = os.path.getsize(full)
            print(f'{prefix}{connector}{entry} ({size} bytes)')
        else:
            print(f'{prefix}{connector}{entry}/')
            extension = '    ' if i == len(entries)-1 else '│   '
            tree(full, prefix + extension)

print('F1-Predictor/')
tree('.')
