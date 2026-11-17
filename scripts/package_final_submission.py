"""Merge final PDFs and package canonical evidence; never commits or pushes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from pypdf import PdfReader,PdfWriter
from pypdf.generic import NameObject,ArrayObject

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.io import atomic_text,file_hash,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-evidence-zip',action='store_true',help='Only merge PDFs and refresh checksums.')
    args=parser.parse_args()
    report=ROOT/'reports/final';artifact=ROOT/'artifacts/final'
    pdfs=[report/'paper.pdf',report/'supplement.pdf']
    if not all(p.exists() for p in pdfs):raise FileNotFoundError('Compile paper.pdf and supplement.pdf first')
    writer=PdfWriter()
    for path in pdfs:
        reader=PdfReader(path)
        # Resolve links before merging: the two documents reuse section names.
        # Explicit page destinations prevent a supplement TOC link from jumping
        # to a same-named destination in the main paper.
        destinations=reader.named_destinations
        for page in reader.pages:
            for reference in page.get('/Annots',[]):
                annotation=reference.get_object()
                action=annotation.get('/A')
                if action is not None:
                    action=action.get_object();name=action.get('/D')
                    if action.get('/S')=='/GoTo' and isinstance(name,str) and name in destinations:
                        action[NameObject('/D')]=ArrayObject(destinations[name].dest_array)
                name=annotation.get('/Dest')
                if isinstance(name,str) and name in destinations:
                    annotation[NameObject('/Dest')]=ArrayObject(destinations[name].dest_array)
        writer.append(reader)
    writer.add_metadata({'/Title':'SAFE-Judge: Shared Adaptive Feedback for Escalation in Relevance Assessment','/Author':'Ayush Ahuja','/Subject':'ACM-style paper and experimental summary'})
    target=ROOT/'reports/stage3_report.pdf'
    with target.open('wb') as f:writer.write(f)
    writer.close()
    checks=[p for p in sorted(artifact.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt']
    atomic_text(artifact/'SHA256SUMS.txt',''.join(f'{file_hash(p)}  {p.relative_to(artifact).as_posix()}\n' for p in checks))
    receipt=dict(paper_pages=len(PdfReader(pdfs[0]).pages),supplement_pages=len(PdfReader(pdfs[1]).pages),combined_pages=len(PdfReader(target).pages),
        paper_sha256=file_hash(pdfs[0]),supplement_sha256=file_hash(pdfs[1]),combined_sha256=file_hash(target),
        compact_artifact_files=len(checks),compact_artifact_bytes=sum(p.stat().st_size for p in checks),
        package_driver_sha256=file_hash(Path(__file__)))
    if not args.skip_evidence_zip:
        files={}
        def include(path,name):
            if path.is_file():
                if name in files and files[name]!=path:raise ValueError('Duplicate ZIP member')
                files[name]=path
        # All compact tables and report sources are directly inspectable.
        for folder in ('artifacts','reports/final'):
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and p.suffix not in ('.log','.blg','.xdv','.png','.aux','.out','.toc','.bbl','.fls') and p.name not in ('submission_manifest.json','visual_qa.json'):
                    include(p,p.relative_to(ROOT).as_posix())
        for p in [target,ROOT/'README.md',ROOT/'reports/ai_usage_statement.md',ROOT/'DATA_NOTICE.md',ROOT/'CITATION.cff']:
            include(p,p.relative_to(ROOT).as_posix())
        for folder in ('src','scripts','configs','tests','docs','notebooks'):
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',) and p.name not in ('colab.yaml','hpc.yaml'):include(p,p.relative_to(ROOT).as_posix())
        for p in ROOT.glob('requirements*.txt'):include(p,p.name)
        # Replay inputs: current COVID only, DL2020 only, and the pilot.
        for name in ('qwen35_covid_proposal_v1_dataset_full','qwen35_transfer_v1_dataset_full'):
            base=ROOT/'results/transfer'/name
            dataset='trec_covid' if 'covid_proposal' in name else 'dl2020'
            for p in (base/'inputs').glob(dataset+'.*'):include(p,'replay_inputs/'+p.relative_to(ROOT).as_posix())
            for p in (base/'runs'/dataset).rglob('*'):
                if p.is_file() and '.sqlite3' not in p.name:include(p,'replay_inputs/'+p.relative_to(ROOT).as_posix())
            for p in (base/'source').rglob('*'):
                if p.is_file():include(p,'inference_frozen_sources/'+name+'/'+p.relative_to(base/'source').as_posix())
            include(base/'archive_identity.json','inference_receipts/'+name+'/archive_identity.json')
            if dataset=='trec_covid':
                for folder in ('source','provenance'):
                    for p in (base/folder).rglob('*'):
                        if p.is_file():include(p,'replay_inputs/'+p.relative_to(ROOT).as_posix())
                include(base/'inputs/suite.json','replay_inputs/'+(base/'inputs/suite.json').relative_to(ROOT).as_posix())
        base=ROOT/'results/stage2/dl2019_qwen35_full_v1'
        for p in base.rglob('*'):
            if p.is_file() and '.sqlite3' not in p.name:include(p,'replay_inputs/'+p.relative_to(ROOT).as_posix())
        # All final acquisition ledgers, plus final CPU-study designs and inputs.
        for base in (ROOT/'results/experiments/covid_proposal_comparison_v1',ROOT/'results/experiments/final_supplement_v1'):
            for p in base.rglob('*'):
                if p.is_file():include(p,'replay_inputs/'+p.relative_to(ROOT).as_posix())
        base=ROOT/'results/experiments/qwen_transfer_comparison_v1'
        for p in (base/'allocations/dl2020_all').rglob('*.gz'):include(p,'replay_inputs/'+p.relative_to(ROOT).as_posix())
        # Detailed in-scope alternative-method logs remain available even
        # though their mixed exploratory parent folders were archived.
        history=ROOT.parent/'SAFE-Judge-local-archive/2026-11-17-final-submission/archived/results/experiments'
        for rel in ('class_threshold_v1/allocations/dl2020_source','class_threshold_v1/calibration/dl2020_source',
                    'class_threshold_v1/reserved_lara/dl2020_all','class_aware_review_v1/allocations/dl2020_all',
                    'gaussian_grade_v1/fits/qwen_dl2020','sparse_query_crc_dl2019_v1','target_pool_sparse_crc_dl2019_v1'):
            for p in (history/rel).rglob('*'):
                if p.is_file():include(p,'historical_in_scope_logs/'+p.relative_to(history).as_posix())
        for family in ('lexical','cross_encoder','lara'):
            base=ROOT/f'results/baselines/{family}/dl2019_shared_test_v1'
            for p in base.rglob('*'):
                if p.is_file() and (p.parent.name=='budget_003664' or p.name in ('split.json','experiment.json','calibration_metrics.csv','temperature.json')):
                    include(p,'historical_in_scope_logs/baselines/'+p.relative_to(ROOT/'results/baselines').as_posix())
        # Preserve the source identities in historical receipts while keeping
        # one-time development utilities outside the publication repository.
        cleanup=report/'code_cleanup_manifest.json'
        if cleanup.exists():
            provenance=json.loads(cleanup.read_text(encoding='utf-8'))
            archive=Path(provenance['archive_root'])
            for action in provenance['actions']:
                source=action['source']
                path=archive/('before_rename' if action['action']=='rename' else 'archived')/source
                include(path,'historical_sources/'+source)
        # Canonical pilot texts and the common baseline split are required by the CPU supplement.
        for p in (ROOT/'data/pilot').glob('*'):
            if p.is_file():include(p,'replay_inputs/'+p.relative_to(ROOT).as_posix())
        include(ROOT/'results/baselines/cross_encoder/dl2019_shared_test_v1/split.json','replay_inputs/results/baselines/cross_encoder/dl2019_shared_test_v1/split.json')
        # Original export hashes identify the complete download; SQLite resume
        # databases are deliberately omitted from the publication package.
        manifest={name:dict(bytes=p.stat().st_size,sha256=file_hash(p)) for name,p in sorted(files.items())}
        destination=ROOT/'dist/safe_judge_final_evidence.zip';destination.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name,p in sorted(files.items()):z.write(p,name)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        with zipfile.ZipFile(destination) as z:
            if z.testzip() is not None:raise ValueError('Corrupt evidence ZIP')
        receipt.update(evidence_zip=destination.relative_to(ROOT).as_posix(),evidence_zip_bytes=destination.stat().st_size,
            evidence_zip_sha256=file_hash(destination),evidence_zip_members=len(manifest)+1,
            evidence_scope='Current COVID, DL2020 and pilot exact inference inputs/receipts; final ledgers; curated in-scope historical tables; frozen inference sources; no model weights')
    write_json(report/'submission_manifest.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
