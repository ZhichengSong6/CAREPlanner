"""Paired rates and x-cluster bootstrap. No old-set R1-count promotion gate."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
from common import ARMS,require,read,jsonl,sha,write,stage_complete


def bootstrap_table(matrix,draws):
    """Rows are ALL predeclared spatial IDs; columns N,R1-pass,new-pass."""
    m=np.asarray(matrix,dtype=float);base=m.sum(axis=0)
    if base[0]==0:return {'n':0,'r1_rate':None,'new_rate':None,'delta_pp':None,'x_clusters_with_cases':0,'ci95_delta_pp':None}
    totals=draws@m;keep=totals[:,0]>0
    d=100*(totals[keep,2]-totals[keep,1])/totals[keep,0]
    return {'n':int(base[0]),'R1':int(base[1]),'SOLVER_GOAL':int(base[2]),
      'r1_rate':base[1]/base[0],'new_rate':base[2]/base[0],
      'delta_pp':100*(base[2]-base[1])/base[0],
      'x_clusters_with_cases':int((m[:,0]>0).sum()),
      'ci95_delta_pp':np.quantile(d,[.025,.975]).tolist() if len(d) else None,
      'bootstrap_valid_draws':int(keep.sum())}


def verify_rows(rows,starts):
    require(len(rows)==len(starts),'Missing evaluation cases')
    require([r.get('case_id') for r in rows]==list(range(len(starts))),'Duplicate/missing case IDs')
    for got,want in zip(rows,starts):
        require(got.get('start')==want,f'Frozen start changed: case {want["case_id"]}')
        for arm in ARMS:
            m=got.get('models',{}).get(arm);require(isinstance(m,dict),f'Missing arm {arm}')
            q=np.asarray(m.get('q_candidate'),float)
            require(q.shape==(7,) and np.isfinite(q).all(),'Nonfinite candidate')
            g=m.get('candidate_g_m');require(isinstance(g,(int,float)) and np.isfinite(g),'Nonfinite FOV')
            require(type(m.get('fov_pass')) is bool and type(m.get('candidate_in_limits')) is bool,'Invalid pass flag')
            require(m['fov_pass']==(m['candidate_in_limits'] and g>=0),'FOV/pass mismatch')
            require(isinstance(m.get('initial_g_m'),(int,float)) and abs(m['initial_g_m']-want['initial_g_m'])<=5e-6,'Initial geometry changed')
        require(abs(got['models']['R1']['initial_g_m']-got['models']['SOLVER_GOAL']['initial_g_m'])<=2e-6,'Arms see different initial geometry')


def summarize(rows,coverage,cfg):
    npoints=cfg['points'];world=cfg['world']
    keys=[(r['point_id'],r['sensor']) for r in coverage]
    require(len(keys)==npoints*8 and len(set(keys))==npoints*8,'Incomplete geometry coverage')
    require(set(keys)=={(i,s) for i in range(npoints) for s in range(8)},'Geometry IDs invalid')
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    # Multinomial cluster counts are equivalent to sampling spatial IDs with replacement.
    draws=rng.multinomial(npoints,np.full(npoints,1/npoints),size=cfg['bootstrap_repetitions'])
    result={}
    for label,chosen in [('all',rows),('local',[r for r in rows if r['start']['cohort']=='local']),('uniform',[r for r in rows if r['start']['cohort']=='uniform'])]+[(f'S{s}',[r for r in rows if r['start']['sensor']==s]) for s in range(8)]:
        matrix=np.zeros((npoints,3));gain=loss=both=neither=0
        margins={str(t):np.zeros((npoints,3)) for t in cfg['margin_thresholds_m']}
        for r in chosen:
            pid=r['start']['point_id'];a=r['models']['R1'];b=r['models']['SOLVER_GOAL']
            old=bool(a['fov_pass']);new=bool(b['fov_pass'])
            matrix[pid]+=[1,old,new]
            gain+=new and not old;loss+=old and not new;both+=old and new;neither+=not old and not new
            for t in cfg['margin_thresholds_m']:
                margins[str(t)][pid]+=[1,a['candidate_in_limits'] and a['candidate_g_m']>=t,b['candidate_in_limits'] and b['candidate_g_m']>=t]
        result[label]=bootstrap_table(matrix,draws)
        result[label]['paired']={'gained':gain,'lost':loss,'both_pass':both,'both_fail':neither,'net':gain-loss}
        result[label]['margin_sensitivity']={t:bootstrap_table(m,draws) for t,m in margins.items()}
    calls={}
    for arm in ARMS:
        calls[arm]={}
        for key in ('solver_ms','value_calls','gradient_calls'):
            v=np.asarray([r['models'][arm].get(key,np.nan) for r in rows],float)
            require(np.isfinite(v).all(),f'Missing timing/call diagnostic: {key}')
            calls[arm][key]={'mean':float(v.mean()) if len(v) else None,'p50':float(np.median(v)) if len(v) else None,'p95':float(np.quantile(v,.95)) if len(v) else None}
    statuses=Counter(c['status'] for c in coverage)
    result.update(status='COMPLETE',scientific_status='REVIEW_REQUIRED',coverage={
        'fixed_x':npoints,'x_with_cases':len({r['start']['point_id'] for r in rows}),
        'x_sensor_pairs':len(coverage),'pair_statuses':dict(statuses),
        'uniform_available':sum(c['uniform_count'] for c in coverage),
        'local_available':sum(c['local_count'] for c in coverage),
        'local_exclusions':sum(len(c['local_exclusions']) for c in coverage),
        'per_sensor':{f'S{s}':dict(Counter(c['status'] for c in coverage if c['sensor']==s)) for s in range(8)}},
        calls_and_timing=calls,failures={a:dict(Counter(r['models'][a]['failure_stage'] for r in rows)) for a in ARMS},
        uncertainty={'method':'percentile paired bootstrap of spatial x clusters; all selected x included, even zero-case x','repetitions':cfg['bootstrap_repetitions'],'seed':cfg['bootstrap_seed'],'level':.95,'scope':'evaluation sampling uncertainty, NOT training-seed variability; CIs per sensor are descriptive, not multiplicity-corrected'},
        scope='OFF-GRID interpolation inside original workspace. WITNESS-conditioned bounded-search population. FOV-only, not LOS/collision/trajectory certification.',
        checkpoints_fixed=True,solver_retuned=False,training_updates=0)
    return result


def markdown(r):
    def pct(x):return 'NA' if x is None else f'{x*100:.2f}%'
    def f(x):return 'NA' if x is None else f'{x:+.2f}'
    lines=['# 新空间点：R1 vs Solver-Goal（相同完整 solver）','',
      '**独立非网格位置、输入先冻结；不训练，不扫阈值。**','',
      '| 分组 | N | R1 | Solver-Goal | Δ pp | 按 x 分组的 95% 区间（pp） | 新增/丢失 |',
      '|---|---:|---:|---:|---:|---|---|']
    for label in ('local','uniform','all')+tuple(f'S{s}' for s in range(8)):
        v=r[label];ci=v['ci95_delta_pp'];p=v['paired'];interval='NA' if ci is None else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'
        lines.append(f'| {label} | {v["n"]} | {pct(v["r1_rate"])} | {pct(v["new_rate"])} | {f(v["delta_pp"])} | {interval} | {p["gained"]}/{p["lost"]} |')
    lines+=['','## 覆盖与限制','',f'预先固定 x={r["coverage"]["fixed_x"]}；有测试起点的 x={r["coverage"]["x_with_cases"]}。',
      f'全部 x-sensor 检索状态：{r["coverage"]["pair_statuses"]}',
      'UNRESOLVED_WITNESS：未找到可见见证的组合没有从点集合删除，也没有替换；不把它们标成不可行。主成功率条件于规定预算内已确认见证的组合。',
      '新集合上的 R1 成功数重新计算，不要求复现旧集合 1291/1963。不同集合的绝对总成功率不能直接比较。','',
      '## 总体候选解析裕量敏感性（不改 solver）','',
      '| FOV 裕量 m | R1 | Solver-Goal | Δ pp |','|---|---:|---:|---:|']
    for t,v in r['all']['margin_sensitivity'].items():lines.append(f'| {t} | {pct(v["r1_rate"])} | {pct(v["new_rate"])} | {f(v["delta_pp"])} |')
    lines+=['','## 计时和调用量','']
    for a,v in r['calls_and_timing'].items():lines.append(f'- {a}: {v}')
    lines+=['','## 解释边界','',
      '区间按空间点重采样，不能当成多训练随机种子的稳定性证明。样本构造只用解析几何，网络在 inputs_frozen.json 写入后才加载。',
      '这是原工作空间内的非网格插值测试，不是新机器人、外推测试或全局精确距离认证。FOV 通过不等于无遮挡、无碰撞、轨迹可执行或实际被观测。',
      '不根据这批结果更换 checkpoint、拼接分支或修改 solver。保留所有失败、未确认组合与逐案例明细。']
    return '\n'.join(lines)+'\n'


def merge(job):
    out=Path(job['out']);stage_complete(out,'inputs_frozen.json')
    starts=list(jsonl(out/'starts.jsonl'));coverage=read(out/'coverage.json')['rows'];rows=[]
    for rank in range(job['cfg']['world']):
        folder=out/'evaluation'/f'rank{rank}';stage_complete(folder,'complete.json')
        rows.extend(jsonl(folder/'solves.jsonl'))
    rows.sort(key=lambda r:r['case_id']);verify_rows(rows,starts)
    r=summarize(rows,coverage,job['cfg'])
    r.update(starts_sha256=sha(out/'starts.jsonl'),points_sha256=job['points_sha256'],checkpoints=job['checkpoint_sha256'])
    if (out/'report.json').exists():require(read(out/'report.json')==r,'Existing report differs; refuse overwrite')
    else:write(out/'report.json',r)
    summary=out/'summary.md'
    if summary.exists():require(summary.read_text(encoding='utf-8')==markdown(r),'Existing summary differs')
    else:summary.write_text(markdown(r),encoding='utf-8')
    write(out/'complete.json',{'status':'COMPLETE','scientific_status':'REVIEW_REQUIRED','case_count':len(rows),
        'files':{'report.json':sha(out/'report.json'),'summary.md':sha(summary)}})
    print(markdown(r),flush=True)
