"""Execute every cell and preserve completed output even after an error."""
from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
os.chdir(root)
path = root / 'experiments.ipynb'
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
# A globally registered python3 kernelspec may otherwise bypass an active venv.
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(notebook, timeout=3600, km=manager, kernel_name='python3',
                        resources={'metadata': {'path': str(root)}}, record_timing=True)

def progress(cell_index, **kwargs):
    cell = notebook.cells[cell_index]
    if cell.cell_type == 'code':
        print(f'Executing cell {cell_index}: {cell.source.splitlines()[0][:100]}', flush=True)

client.on_cell_start = progress
client.on_cell_executed = lambda **kwargs: nbformat.write(notebook, path)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
    print(f'Saved {path}', flush=True)
print('All notebook cells executed successfully.', flush=True)
