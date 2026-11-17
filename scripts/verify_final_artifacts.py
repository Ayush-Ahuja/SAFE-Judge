"""Verify released checksums, label contracts and final budget arithmetic."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys
import zipfile

import numpy as np
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.io import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,help='Additionally verify every member against PACKAGE_MANIFEST.json.')
    args=parser.parse_args()
    data=ROOT/'artifacts/final';checked=0
    for line in (data/'SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split('  ',1);p=(data/name).resolve()
        if not p.is_relative_to(data.resolve()) or file_hash(p)!=digest:raise AssertionError(name)
        checked+=1
    for dataset,cutoff in [('dl2019',2),('dl2020',2),('covid_reserved',1),('covid_all',1)]:
        with (data/f'{dataset}_predictions.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        assert len({(r['query_id'],r['doc_id']) for r in rows})==len(rows)
        assert all(int(r['true_binary'])==int(int(r['true_relevance'])>=cutoff) for r in rows)
    with (data/'policy_runs.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        n,h,a,e=map(int,(r['pairs'],r['human_labels'],r['automatic_labels'],r['automatic_errors']))
        assert h+a==n and h==int(r['budget'])
        assert np.isclose(float(r['auto_error_rate']),e/n)
        if a:assert np.isclose(float(r['selective_risk']),e/a)
    build=json.loads((ROOT/'reports/final/build_validation.json').read_text())
    assert build['builder_sha256']==file_hash(ROOT/'scripts/build_final_report.py')
    for name,digest in build['input_sha256'].items():assert file_hash(data/name)==digest,name
    pdfs={}
    for name in ('paper','supplement'):
        path=ROOT/f'reports/final/{name}.pdf';p=PdfReader(path)
        pdfs[name]=len(p.pages)
        text='\n'.join(page.extract_text() for page in p.pages)
        assert 'ChatReport' not in text
        assert re.search(r'\b(student|instructor)\b',text,re.IGNORECASE) is None
    members=0
    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
            assert set(z.namelist())==set(manifest)|{'PACKAGE_MANIFEST.json'}
            assert not any('.sqlite3' in name for name in manifest)
            for name,entry in manifest.items():
                digest=hashlib.sha256();length=0
                with z.open(name) as f:
                    for block in iter(lambda:f.read(1024*1024),b''):digest.update(block);length+=len(block)
                assert length==entry['bytes'] and digest.hexdigest()==entry['sha256'],name
            members=len(manifest)
    print(json.dumps(dict(status='verified',artifact_checksums=checked,policy_rows=len(rows),pdf_pages=pdfs,zip_members_hashed=members),indent=2))


if __name__=='__main__':main()
