"""Package the canonical grade>=1 COVID GPU workflow from frozen run sources.

Uses the completed export's exact inference implementation and manifest, not
the development-time multi-collection upload bundle. The notebook's prose is
current; all scoring cells retain the completed workflow.
"""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.transfer_colab_support import safe_extract


def encoded(value):
    return (json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode('utf-8')


def main():
    source=ROOT/'results/transfer/qwen35_covid_proposal_v1_dataset_full'
    if not (source/'source/src/assessor/transfer_prompts.py').exists():
        archive=ROOT/'dist/qwen35_covid_proposal_v1_dataset_full_results.zip'
        if not archive.exists():raise FileNotFoundError('Extract the canonical COVID inference export from the shared evidence package first')
        safe_extract(archive,source)
    payload={}
    for p in (source/'source').rglob('*'):
        if p.is_file() and p.suffix=='.py' and p.name!='build_proposal_covid_colab.py':payload[p.relative_to(source/'source').as_posix()]=p.read_bytes()
    meta=json.loads((source/'inputs/trec_covid.metadata.json').read_text())
    if meta['relevance_threshold']!=1:raise ValueError('Canonical COVID relevance cutoff must be 1')
    manifest=(source/'inputs/trec_covid.jsonl').read_bytes()
    if hashlib.sha256(manifest).hexdigest()!=meta['manifest_sha256']:raise ValueError('Manifest hash mismatch')
    notebook=json.loads((ROOT/'notebooks/qwen_covid_proposal_colab.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(cell['source'],cell.get('id','notebook'),'exec')
    payload['notebooks/qwen_covid_proposal_colab.ipynb']=encoded(notebook)
    payload['data/transfer/trec_covid.jsonl']=manifest
    payload['data/transfer/trec_covid.metadata.json']=encoded(meta)
    payload['data/transfer/suite.json']=encoded({'schema_version':1,'datasets':{'trec_covid':{'manifest':'data/transfer/trec_covid.jsonl',**meta}}})
    payload['configs/default.yaml']=(source/'provenance/configs/transfer_colab.yaml').read_bytes()
    payload['scripts/build_proposal_covid_colab.py']=Path(__file__).read_bytes()
    payload['docs/qwen_transfer_colab.md']=b'# COVID GPU inference\n\nUse a GPU runtime and run the notebook. Grades 1 and 2 are relevant. The 5000-pair sample and 512-token total input cap match the final report.\n'
    payload['bundle_hashes.json']=encoded({name:hashlib.sha256(data).hexdigest() for name,data in sorted(payload.items())})
    target=ROOT/'dist/qwen_covid_proposal_colab.zip';target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):archive.writestr(name,data)
    print(target,f'{target.stat().st_size/2**20:.2f} MiB')


if __name__=='__main__':main()
