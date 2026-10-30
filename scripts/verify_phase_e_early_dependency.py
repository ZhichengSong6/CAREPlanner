#!/usr/bin/env python3
"""Compare actual old/new production objective blocks on frozen CASE020 rows."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from test_phase_e_dependency_objective import PREFIX

def block(source):
    start = source.index('  if (repair_mode &&\n      frontier.active &&')
    end = source.index('  for (int k = 0; k < num_intervals_; ++k)', start)
    return source[start:end], source[:start]+source[end:]

def cpp_fixture(ctx, source_block):
    prefix = PREFIX[:PREFIX.index('Result objective(')]
    initialization = '''Result objective() {
      const bool repair_mode=true; const int dof_=7,num_intervals_=20,visibility_frontier_horizon_step_=2;
      const double visibility_waypoint_weight_=3000.,trust_radius=.15;
      struct Frontier { bool active=true; V q,joint_mask,recovery_target; int recovery_attempt=0;
        double frontier_weight_scale=1.; } frontier;
      frontier.q=V(7); frontier.joint_mask=V(7);
      Eigen::MatrixXd q_bar=Eigen::MatrixXd::Zero(7,21);
      std::vector<Row> selected;
      care_collision_cdf::CollisionCDFConstraintBatch batch;
      Result out;
      auto finiteVector=[](const V& q){return q.allFinite();};
      auto qIndex=[](int k,int j){return (k-1)*7+j;};
      auto addQuadraticTarget=[&](int i,double w,double q){out.terms.emplace_back(i,w,q);};
    '''
    for j in range(7):
        initialization += f'frontier.q[{j}]={ctx["target"][j]:.17g}; frontier.joint_mask[{j}]={ctx["mask"][j]:.17g};\n'
        for k in range(21): initialization += f'q_bar({j},{k})={ctx["qbar"][k][j]:.17g};\n'
    for i, row in enumerate(ctx['rows']):
        initialization += f'{{ Row r; r.pair={i};r.k={row["k"]};r.d={row["d"]:.17g};r.safety_margin=.05;\n'
        for j in range(7):initialization += f'r.g[{j}]={row["g"][j]:.17g};r.qlin[{j}]={row["qlin"][j]:.17g};\n'
        initialization += 'selected.push_back(r);batch.source_type.push_back(0);'
        for v in row['point']:initialization += f'batch.point_flat.push_back({v:.17g});'
        initialization += '}\n'
    suffix = r'''return out; }
    int main(){auto out=objective();std::cout.precision(17);
      std::cout<<"{\"dependencies\":[";bool first=true;
      for(const auto&p:out.observation_dependency_points){if(!first)std::cout<<',';first=false;
        std::cout<<'['<<p[0]<<','<<p[1]<<','<<p[2]<<']';}
      std::cout<<"],\"terms\":[";first=true;
      for(const auto&t:out.terms){if(!first)std::cout<<',';first=false;
        std::cout<<'['<<std::get<0>(t)<<','<<std::get<1>(t)<<','<<std::get<2>(t)<<']';}
      std::cout<<"]}\n";}
    '''
    return prefix+initialization+source_block+suffix

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    cpp=ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp'
    sources={'before':args.before.read_text(),'after':cpp.read_text()}
    sections={k:block(v) for k,v in sources.items()}
    assert sections['before'][1]==sections['after'][1], 'Change outside objective block'
    source=ROOT/'outputs/phase_e_case020_observation_analysis_20261011/constraint_attribution'
    contexts=json.loads((source/'contexts.json').read_text());cases=[]
    for ctx in contexts:
        outputs={}
        for name,(section,_) in sections.items():
            stem=out/(str(ctx['stamp'])+'_'+name);path=stem.with_suffix('.cpp')
            path.write_text(cpp_fixture(ctx,section))
            build=subprocess.run(['g++','-std=c++14','-O0','-I/usr/include/eigen3',
                '-I'+str(ROOT/'src/egocentric_arm_planner/include'),str(path),'-o',str(stem)],capture_output=True,text=True)
            stem.with_suffix('.build.log').write_text(build.stdout+build.stderr);build.check_returncode()
            outputs[name]=json.loads(subprocess.check_output([str(stem)],text=True))
        assert outputs['before']['terms']==outputs['after']['terms']
        assert not outputs['before']['dependencies']
        expected=[[.15,.1,.45]] if ctx['stamp']==20001000000 else [[.15,.15,.55],[.15,.1,.45]]
        actual=outputs['after']['dependencies'];assert len(actual)==len(expected)
        assert all(any(max(abs(a-b) for a,b in zip(p,q))<1e-5 for q in actual) for p in expected)
        cases.append(dict(query_stamp_ns=ctx['stamp'],before=outputs['before'],after=outputs['after'],objective_equal=True,
                          original_cdf_rows=len(ctx['rows'])))
    fixtures=json.loads((source/'fixtures.json').read_text())
    fixtures=[f for f in fixtures if f['name'].endswith('_baseline')]
    (out/'qp_inputs.json').write_text(json.dumps(fixtures))
    subprocess.run([str(source/'solve'),str(out/'qp_inputs.json'),str(out/'qp_results.json')],check=True)
    original={r['name']:r for r in json.loads((source/'results.json').read_text())}
    errors=[]
    for r in json.loads((out/'qp_results.json').read_text()):
        assert r['status']=='solved';error=max(abs(a-b) for a,b in zip(r['x'],original[r['name']]['x']))
        assert error<1e-10;errors.append(error)
    report=dict(status='PASS',cases=cases,all_other_qp_assembly_source_identical=True,
        same_objective_and_constraints_replayed_qp_max_errors=errors,
        qualification='Compiles actual production objective blocks and proves all other QP assembly source identical; reruns frozen original QPs with the same vendored PIQP. This is local dependency detection, not proof of globally blocked motion.',
        sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',cases=[dict(stamp=c['query_stamp_ns'],dependencies=c['after']['dependencies']) for c in cases],qp_errors=errors)))

if __name__=='__main__':main()
