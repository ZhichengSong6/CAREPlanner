"""User-authorized CDF checkpoint comparison for declared original tasks."""
from pathlib import Path
import hashlib,json,os
T=Path(__file__).resolve().parent;RT=T.parents[1];O=RT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_old_diagnostic(batch,ids,seconds):
    assert not (O.parent/'artifact_sha256.json').exists(),'Study sealed'
    plan=json.loads((O/'plan.json').read_text());matches=[r for r in plan['runs'] if r['batch']==batch]
    assert len(matches)==1 and ids==['phase_e_goal_'+matches[0]['case']] and seconds==matches[0]['task_wall_budget_s'] and seconds in (60,120)
    item=matches[0];scene=json.loads((O/'scenes.json').read_text())['configurations'][item['configuration']]
    assert os.environ['WORLD_FILE']==scene['world'] and sha(Path(scene['world']))==scene['world_sha256']
    assert os.environ['CARE_STATIC_SCENE_ID']==item['configuration']
    path=O/('new_model_binding.json' if item['model']=='new' else 'old_model_binding.json')
    assert os.environ['CARE_CDF_MODEL_BINDING']==str(path)
    binding=json.loads(path.read_text());assert sha(Path(binding['checkpoint']))==binding['checkpoint_sha256']
    expected={'old':('41a4228eee797971d56b46849e0f52c663eae79dc35667e8704c1bc3c8d72c70','gelu'),
              'new':('d153c5b08af0d2b8ce12da500fe13d5d0b04cfc0d9ce07a2e6ad4d2dfd93a4e6','relu')}
    assert (binding['checkpoint_sha256'],binding['activation'])==expected[item['model']]
    for n,h in json.loads((O/'frozen_sha256.json').read_text()).items():assert sha(O/n)==h,n
    assert os.environ['CARE_SCENE_GUARD_PREFIX']==str(T/(item['tag']+'_guard'))
    return True
