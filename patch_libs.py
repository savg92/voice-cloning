import os
import re

def patch_file(path, pattern, replacement):
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return
    with open(path, 'r') as f:
        content = f.read()
    
    new_content = re.sub(pattern, replacement, content)
    
    if content != new_content:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f"Patched {path}")
    else:
        print(f"No changes needed for {path}")

VENV = '/Users/savg/Desktop/voice-cloning/.venv/lib/python3.12/site-packages'

# 1. Kokoro fixes
patch_file(f'{VENV}/kokoro/istftnet.py', r'from torch\.nn\.utils import weight_norm', 'from torch.nn.utils.parametrizations import weight_norm')
patch_file(f'{VENV}/kokoro/modules.py', r'from torch\.nn\.utils import weight_norm', 'from torch.nn.utils.parametrizations import weight_norm')
patch_file(f'{VENV}/kokoro/modules.py', r'num_layers=1, batch_first=True, bidirectional=True, dropout=dropout', 'num_layers=1, batch_first=True, bidirectional=True, dropout=0')

# 2. Misaki fixes
for fpath in [f'{VENV}/misaki/en.py', f'{VENV}/misaki/cutlet.py']:
    patch_file(fpath, r'importlib\.resources\.open_text\(data, (.*?)\)', r'importlib.resources.files(data).joinpath(\1).open()')

# 3. SpaCy & Weasel (Click deprecation)
for fpath in [f'{VENV}/spacy/cli/_util.py', f'{VENV}/weasel/util/config.py']:
    patch_file(fpath, r'from click\.parser import split_arg_string', 'from click.shell_completion import split_arg_string')

# 4. Jieba (pkg_resources)
patch_file(f'{VENV}/jieba/_compat.py', r'import pkg_resources', 'import importlib.metadata as pkg_resources')

# 5. PyAnnote (namespace deprecation - silencing is the only way for legacy declare_namespace)
# But we can patch the files calling it to avoid the warning if we find them.
# For now, let's at least fix the ones we specifically found in the trace.

print("Patching complete!")
