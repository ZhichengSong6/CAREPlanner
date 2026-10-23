"""All-case budget curves. Never replace failures by successful-only averages."""
from __future__ import annotations
import csv
import os
from pathlib import Path
import tempfile
import numpy as np
from ie_common import ARMS, require, read, lines, sha, write, finish, stage


def stats(values):
    a = np.asarray(values, dtype=float)
    require(np.isfinite(a).all(), 'Nonfinite statistic; do not silently omit failures')
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.median(a)), p95=float(np.quantile(a, .95))) if len(a) else dict(n=0, mean=None, p50=None, p95=None)


def comparison(part, budget, cfg):
    n = len(part)
    if not n:
        return dict(n=0, gained=0, lost=0, delta_pp=None, cluster_x_95_pp=None)
    old = np.array([r['models']['R1'][budget]['fov_pass'] for r in part], dtype=np.int8)
    new = np.array([r['models']['B_keep'][budget]['fov_pass'] for r in part], dtype=np.int8)
    pids = np.array([r['point_id'] for r in part], dtype=int)
    denominator = np.bincount(pids, minlength=cfg['point_count'])
    difference = np.bincount(pids, weights=new-old, minlength=cfg['point_count'])
    rng = np.random.default_rng(cfg['bootstrap_seed']); boot = []
    for first in range(0, cfg['bootstrap_repeats'], 128):
        count = min(128, cfg['bootstrap_repeats']-first)
        ix = rng.integers(cfg['point_count'], size=(count, cfg['point_count']))
        den = denominator[ix].sum(1); num = difference[ix].sum(1)
        boot.extend((100*num[den > 0]/den[den > 0]).tolist())
    return dict(n=n, gained=int(((new == 1) & (old == 0)).sum()),
                lost=int(((old == 1) & (new == 0)).sum()),
                delta_pp=float(100*(new-old).mean()),
                cluster_x_95_pp=np.quantile(boot, [.025, .975]).tolist() if boot else None)


def summarize(part, budget, cfg):
    result = dict(n=len(part), paired=comparison(part, budget, cfg), models={})
    for name in ARMS:
        a = [r['models'][name][budget] for r in part]
        success = sum(r['fov_pass'] for r in a)
        result['models'][name] = dict(n=len(a), success=success, failure=len(a)-success,
            success_rate=success/len(a) if a else None,
            solver_ms=stats([r['solver_ms'] for r in a]),
            gradient_calls=stats([r['gradient_calls'] for r in a]),
            value_calls=stats([r['value_calls'] for r in a]),
            root_value_calls=stats([r['root_value_calls'] for r in a]),
            violation_m=stats([r['violation_m'] for r in a]),
            noops=sum(r['noop_calls'] for r in a),
            budget_exhausted=sum(r['termination'] == 'gradient_budget_exhausted' for r in a),
            margin_rates={str(t):sum(r['candidate_in_limits'] and r['candidate_g_m'] >= t for r in a)/len(a) if a else None
                          for t in cfg['margin_thresholds_m']})
    return result


def validate_rows(rows, starts, refs, job):
    from evaluate import check_full
    cfg = job['cfg']; keys = {str(k) for k in cfg['budgets']} | {'full'}
    require([r['case_id'] for r in rows] == list(range(len(starts))), 'Missing/duplicated evaluation cases')
    for row, start, ref in zip(rows, starts, refs):
        require(all(row[k] == start[k] for k in ('case_id', 'point_id', 'cohort', 'sensor')), 'Input identity mismatch')
        require(set(row['models']) == set(ARMS), 'Missing model result')
        for name in ARMS:
            require(set(row['models'][name]) == keys, 'Missing budget result')
            for budget, data in row['models'][name].items():
                require(type(data['fov_pass']) is bool and np.isfinite(data['candidate_g_m']), 'Invalid geometry')
                q = np.asarray(data['q_candidate']); require(q.shape == (7,) and np.isfinite(q).all(), 'Invalid candidate q')
                legal = bool((q >= job['lo']).all() and (q <= job['hi']).all())
                require(legal == data['candidate_in_limits'], 'Limits inconsistent')
                require(data['fov_pass'] == (legal and data['candidate_g_m'] >= 0), 'FOV flag inconsistent')
                require(abs(data['violation_m']-max(0., -data['candidate_g_m'])) < 1e-12, 'Residual inconsistent')
                require(data['gradient_calls'] == data['projection_calls']+data['ascent_calls'], 'Gradient/update accounting mismatch')
                if budget != 'full':
                    require(data['budget'] == int(budget) and data['gradient_calls'] <= int(budget), 'Wrong budget')
                require(data['root_value_calls'] <= data['value_calls'], 'Value-only accounting wrong')
            check_full(row['models'][name]['full'], ref['models'][name])


def merge(job):
    out = Path(job['out']); final = out / 'report'
    if final.exists():
        stage(final); print('[reuse] verified report', flush=True); return
    starts = list(lines(out / 'starts.jsonl')); refs = list(lines(out / 'reference.jsonl')); rows = []
    for rank in range(job['cfg']['world']):
        folder = out / 'evaluation' / f'rank{rank}'; stage(folder)
        rows.extend(lines(folder / 'results.jsonl'))
    rows.sort(key=lambda r:r['case_id']); validate_rows(rows, starts, refs, job)
    cfg = job['cfg']; budgets = [str(k) for k in cfg['budgets']] + ['full']
    groups = {'local': [r for r in rows if r['cohort'] == 'local'],
              'uniform': [r for r in rows if r['cohort'] == 'uniform'], 'all':rows}
    groups.update({f'S{s}':[r for r in rows if r['sensor'] == s] for s in range(8)})
    results = {label:{k:summarize(part, k, cfg) for k in budgets} for label, part in groups.items()}
    hit = {}
    for group in ('local', 'uniform', 'all'):
        hit[group] = {}
        for name in ARMS:
            values = [r['models'][name]['full']['first_hit_gradient_query_diagnostic'] for r in groups[group]]
            found = [v for v in values if v is not None]
            hit[group][name] = dict(n=len(values), no_recorded_hit=len(values)-len(found),
                hit_any_recorded_state=len(found), conditional_on_hit=stats(found),
                meaning='Full-run accepted-state first hit only; not a returned-candidate success curve and not an online oracle policy.')
    tmp = Path(tempfile.mkdtemp(prefix='.report.', dir=out))
    from collections import Counter
    coverage = dict(Counter(r['status'] for r in lines(out / 'coverage.jsonl')))
    payload = dict(status='COMPLETE', protocol=cfg, groups=results, first_hit_diagnostic=hit,
                   coverage=coverage, full_reference_parity='PASS', starts_sha256=sha(out / 'starts.jsonl'))
    write(tmp / 'report.json', payload)
    columns = ['group', 'budget', 'N', 'R1_success', 'B_success', 'R1_rate', 'B_rate', 'delta_pp',
               'ci_lo_pp', 'ci_hi_pp', 'gained', 'lost', 'R1_ms_p50', 'B_ms_p50', 'R1_ms_p95', 'B_ms_p95',
               'R1_violation_mean_m', 'B_violation_mean_m', 'R1_gradient_mean', 'B_gradient_mean',
               'R1_value_mean', 'B_value_mean', 'R1_bisection_value_mean', 'B_bisection_value_mean']
    with (tmp / 'curve.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for group, values in results.items():
            for k, r in values.items():
                a, b, p = r['models']['R1'], r['models']['B_keep'], r['paired']
                ci = p['cluster_x_95_pp'] or [None, None]
                writer.writerow(dict(group=group, budget=k, N=r['n'], R1_success=a['success'], B_success=b['success'],
                    R1_rate=a['success_rate'], B_rate=b['success_rate'], delta_pp=p['delta_pp'], ci_lo_pp=ci[0], ci_hi_pp=ci[1],
                    gained=p['gained'], lost=p['lost'], R1_ms_p50=a['solver_ms']['p50'], B_ms_p50=b['solver_ms']['p50'],
                    R1_ms_p95=a['solver_ms']['p95'], B_ms_p95=b['solver_ms']['p95'],
                    R1_violation_mean_m=a['violation_m']['mean'], B_violation_mean_m=b['violation_m']['mean'],
                    R1_gradient_mean=a['gradient_calls']['mean'], B_gradient_mean=b['gradient_calls']['mean'],
                    R1_value_mean=a['value_calls']['mean'], B_value_mean=b['value_calls']['mean'],
                    R1_bisection_value_mean=a['root_value_calls']['mean'], B_bisection_value_mean=b['root_value_calls']['mean']))
    text = ['# R1 vs B：梯度查询预算—FOV 候选成功率', '',
        'Uniform 是起点分布，不是一步。K 是最多梯度查询/配置更新尝试数；二分细化仍可进行，值查询另记。',
        '每个预算从相同起点独立运行；预算耗尽时返回请求下一梯度时的当前配置，或提前结束的原候选。',
        '**主指标不选轨迹最优点、不删除失败、不把真实 FOV oracle 接入搜索。**',
        '复用已经看过结果的 Keep-v2 512 点确认集：这是冻结收敛特征测试，不是新的未见泛化证明。', '']
    for group in ('local', 'uniform', 'all'):
        text += [f'## {group}', '', '| K | N | R1 | B | B−R1 pp [x-cluster 95%] | R1/B p50 ms | R1/B mean FOV violation m |',
                 '|---|---:|---:|---:|---|---|---|']
        for k, r in results[group].items():
            a, b, p = r['models']['R1'], r['models']['B_keep'], r['paired']; ci = p['cluster_x_95_pp']
            text.append(f"| {k} | {r['n']} | {100*a['success_rate']:.2f}% | {100*b['success_rate']:.2f}% | "
                f"{p['delta_pp']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] | {a['solver_ms']['p50']:.2f}/{b['solver_ms']['p50']:.2f} | "
                f"{a['violation_m']['mean']:.6f}/{b['violation_m']['mean']:.6f} |")
        text.append('')
    text += ['## 解释与复核', '',
        '- full 必须逐案例复现上一轮的候选、成功标志、根来源和查询次数；否则评估直接失败。',
        '- 有限 K 下的返回候选成功率不保证单调；不能事后改成“到过的最好点”。',
        '- report.json 的 first_hit_diagnostic 是完整轨迹上首次经过合格状态的事后统计，不是主成功率。',
        '- 首次成功的平均/分位步数只对有命中者定义；未命中数独立报告，绝不当作0步。',
        '- 计时是带少量记录开销的单次 solver，包含二分及Python开销，不含事后解析检查，也不是单次网络前向。',
        '- CSV 同时报梯度查询、值查询和二分查询；不能把 K=1 描述成只有一次前向计算。',
        '- 依旧是指定单 sensor 的 FOV-only 候选测试，不是 LOS、碰撞、执行或真实观测成功率。',
        f'- 覆盖：{coverage}；未确认组合不当作不可行。',
        '- 空间点 bootstrap 只反映固定权重与此采样分布，不代表多训练种子稳定性。']
    # Optional vector figures; report/CSV remain authoritative if matplotlib is unavailable.
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        for group in ('local', 'uniform'):
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            for name in ARMS:
                ax.plot(cfg['budgets'], [100*results[group][str(k)]['models'][name]['success_rate'] for k in cfg['budgets']],
                        marker='o', label=name)
            ax.set_xlabel('Maximum gradient queries / update attempts')
            ax.set_ylabel('Returned-candidate FOV success (%)'); ax.set_ylim(0, 100)
            ax.set_xticks(cfg['budgets']); ax.set_title(group); ax.legend(); fig.tight_layout()
            fig.savefig(tmp / f'success_{group}.svg'); plt.close(fig)
        text += ['', 'SVG 曲线已生成；图的横轴不包含 full（full 不是固定的梯度次数）。']
    except ImportError:
        text += ['', 'SVG: NOT_RUN (matplotlib unavailable); curve.csv 已完整保存。']
    (tmp / 'summary.md').write_text('\n'.join(text) + '\n')
    finish(tmp, source_sha256=job['code_hashes'], models=job['models'], inputs_sha256=sha(out/'starts.jsonl'))
    os.rename(tmp, final); print('\n'.join(text), flush=True)
