from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient
if __name__=='__main__':
    status={}
    for p in sorted(Path('notebooks').glob('*.ipynb')):
        n=nbformat.read(p,as_version=4)
        NotebookClient(n,timeout=120,kernel_name='python3').execute()
        status[p.name]='passed'
    Path('results/notebook-status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(status)
