"""Record a valid no-motion task failure without inventing execution evidence."""
from pathlib import Path
import argparse,hashlib,json,sys
O=Path(__file__).resolve().parent;R=O/'fixed/runtime';T=R/'outputs/cdf_model_ab'
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--case',required=True);p.add_argument('--batch',required=True);a=p.parse_args()
    result=json.loads((T/(a.tag+'_result.json')).read_text())
    assert result['runner_rc']==0 and result['valid_window'] and result['archive_crc_valid']
    assert not result['task_success'] and not result['strict_success'] and not result['overall_safe']
    assert result['commits']==0 and result['execution_vbc']['records']==0
    assert result['review_rc']==result['cdf_review_rc']==0 and result['analysis_rc']==1
    assert not result['contact_events'] and not result['contact_guard']['tripped']
    source=T/'analyze.py';code=source.read_text();assert "assert execution,'missing execution VBC records'" in code
    code=code.replace("assert execution,'missing execution VBC records'", "assert not execution and not commits, 'Only zero-motion empty-audit case is recoverable'")
    code=code.replace("status='PASS' if not unsafe_execution else 'REVIEW_REQUIRED'", "status='NO_MOTION_NO_EXECUTION_EVIDENCE'")
    sys.argv=[str(source),'--tag',a.tag,'--case',a.case,'--batch',a.batch]
    exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
    path=T/('online_ledger_'+a.tag+'.json');ledger=json.loads(path.read_text())
    ledger['stationary_failure_review']=dict(original_analysis_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qualification='Original task=false and overall_safe=false retained. No committed motion and no execution VBC records. Missing execution data is not relabelled as safe execution.')
    path.write_text(json.dumps(ledger,indent=2)+'\n')
if __name__=='__main__':main()
