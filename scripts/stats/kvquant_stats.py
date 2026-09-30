#!/usr/bin/env python3
"""KV-cache quantization quality claims: statistical re-analysis.

Read-only. Python 3 standard library only. Deterministic.

Usage:
    python3 kvquant_stats.py <inspect_log_dir> <out_dir>

Reads the inspect_ai .eval logs named in MANIFEST (resolved by timestamp
prefix inside <inspect_log_dir>), pairs runs on verified-identical item
ids + content hashes, and writes <out_dir>/results.json and
<out_dir>/results.md.

Methods (all two-sided, alpha = 0.05 unless stated):
  * single proportion: Clopper-Pearson exact (primary), Wilson score;
    GPQA additionally a finite-population (hypergeometric exact) interval,
    labelled "valid only under deterministic decoding".
  * paired comparison: exact McNemar (binomial on discordant pairs b, c),
    Newcombe (1998) method 10 hybrid-score CI for the paired difference.
  * Holm-Bonferroni within a named family.
  * power / minimum detectable effect for the exact McNemar test (exact
    enumeration), and normal-approximation sample sizes for a follow-up.
The statistical primitives live in statlib.py (tested by
tests/python/test_stats_statlib.py).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from math import comb, exp, lgamma, log, sqrt, ceil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvquant_extract_samples as ex  # noqa: E402
from statlib import (ALPHA, Z975, Z80, clopper_pearson, holm,  # noqa: E402
                     hypergeometric_interval, mcnemar_exact,
                     mcnemar_mde, mcnemar_power, n_paired_noninferiority,
                     n_paired_superiority, newcombe_paired, newcombe_unpaired,
                     sign_flip_exact, wilson)



# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

# label -> timestamp prefix of the inspect log file. Serving conditions of
# each run were established from devai-router.log / devai-ollama.log /
# devai-vllm.log (see REPORT.md section 2); they are recorded here as data.
MANIFEST = {
    # qwen3.6:35b-a3b-mtp-q4_K_M, Ollama 0.31.1, flash attention enabled in
    # every runner (llama-server "--flash-attn auto -> enabled" / "on").
    'qwen_f16_64K': {
        'gsm8k': '2026-07-18T18-03-44', 'humaneval': '2026-07-18T18-25-27',
        'humaneval_plus': '2026-07-19T17-38-15', 'mmlu_pro': '2026-07-19T17-50-48',
        'gpqa': '2026-07-19T18-11-33', 'tools': '2026-07-20T14-56-37'},
    'qwen_q8_128K': {
        'gsm8k': '2026-07-21T09-56-11', 'humaneval': '2026-07-21T10-23-54',
        'humaneval_plus': '2026-07-21T10-36-40', 'mmlu_pro': '2026-07-21T10-47-53',
        'gpqa': '2026-07-21T11-12-01', 'tools': '2026-07-21T11-10-52'},
    'qwen_q8_128K_rep2': {'gpqa': '2026-07-21T12-12-16'},
    # f16 at n_ctx 262144 with 41/42 layers on GPU (partial CPU offload):
    # a different serving condition, used only as a test-retest reference.
    'qwen_f16_256K_partialCPU': {
        'gsm8k': '2026-07-18T07-36-35', 'humaneval': '2026-07-18T08-15-57',
        'tools': '2026-07-18T08-40-36'},
    # gemma4:26b-a4b-it-q4_K_M, Ollama 0.31.1, FA enabled in every runner.
    'gemma_f16_128K': {
        'gsm8k': '2026-07-18T10-58-13', 'humaneval': '2026-07-18T11-21-17',
        'humaneval_plus': '2026-07-19T15-48-02', 'mmlu_pro': '2026-07-19T15-57-55',
        'gpqa': '2026-07-19T16-32-40', 'tools': '2026-07-20T14-55-09'},
    'gemma_q8_256K': {
        'gsm8k': '2026-07-21T15-39-25', 'humaneval': '2026-07-21T16-05-18',
        'humaneval_plus': '2026-07-21T16-13-28', 'mmlu_pro': '2026-07-21T16-21-35',
        'gpqa': '2026-07-21T16-59-37', 'tools': '2026-07-21T18-04-23'},
    # f16 at n_ctx 262144, 31/31 layers on GPU: SAME context as gemma_q8_256K.
    'gemma_f16_256K': {
        'gsm8k': '2026-07-17T21-06-31', 'humaneval': '2026-07-17T21-32-03',
        'tools': '2026-07-17T21-41-17'},
    # gpt-oss-20b@131072 on vLLM 0.22.1: fp8 KV (07-17 .. 07-20) vs auto (07-22).
    'gptoss_fp8': {
        'gsm8k': '2026-07-17T19-34-23', 'humaneval': '2026-07-17T19-35-25',
        'humaneval_plus': '2026-07-19T00-52-27', 'mmlu_pro': '2026-07-18T23-59-46',
        'gpqa': '2026-07-19T00-06-12', 'tools': '2026-07-17T19-36-18'},
    'gptoss_fp8_rep2': {'tools': '2026-07-20T14-46-01'},
    'gptoss_auto': {
        'gsm8k': '2026-07-22T07-04-39', 'humaneval': '2026-07-22T07-05-51',
        'humaneval_plus': '2026-07-22T07-06-34', 'mmlu_pro': '2026-07-22T07-07-27',
        'gpqa': '2026-07-22T07-15-22', 'tools': '2026-07-22T07-28-33'},
    'gptoss_auto_rep2': {'humaneval_plus': '2026-07-22T07-29-03'},
}

TASK_OF = {'gsm8k': 'gsm8k_task', 'humaneval': 'humaneval_task',
           'humaneval_plus': 'humaneval_plus_task', 'mmlu_pro': 'mmlu_pro_task',
           'gpqa': 'gpqa_task', 'tools': 'tools_use_task'}

# Sampling frame of each item set (from scripts/bench/tasks/*.py).
FRAME = {
    'gsm8k': 'first 100 items of the GSM8K test split (fixed set, not random)',
    'humaneval': 'first 50 of 164 HumanEval problems (fixed set, not random)',
    'humaneval_plus': 'first 50 of 164 HumanEval+ problems (fixed set, not random)',
    'mmlu_pro': 'seeded (seed 42) shuffle of the MMLU-Pro test split, first n (random sample w/o replacement)',
    'gpqa': 'seeded (seed 42) shuffle of the 198 GPQA-Diamond items, first n (random sample w/o replacement)',
    'tools': 'the 20 hand-written tools_use prompts (the whole population)',
}


def load_run(log_dir: str, prefix: str) -> dict:
    hits = sorted(glob.glob(os.path.join(log_dir, prefix + '*.eval')))
    if len(hits) != 1:
        raise SystemExit(f'expected exactly one log for {prefix}, got {hits}')
    import zipfile
    z = zipfile.ZipFile(hits[0])
    h = json.loads(z.read('header.json'))
    recs = {}
    for nm in z.namelist():
        if not nm.startswith('samples/'):
            continue
        s = json.loads(z.read(nm))
        val, _ans = ex.score_of(s)
        if val in ('C', 'I'):
            y = 1 if val == 'C' else 0
        else:
            y = 1 if float(val) >= 0.5 else 0
        stop, _r, _a, usage = ex.out_info(s)
        recs[s['id']] = {'item': ex.item_hash(s), 'y': y,
                         'timeout': bool(s.get('limit')),
                         'noanswer_other': (not s.get('limit')) and stop not in ('stop', 'tool_calls'),
                         'tokens': usage.get('output_tokens')}
    return {'file': os.path.basename(hits[0]), 'task': h['eval']['task'],
            'model': h['eval']['model'], 'created': h['eval']['created'],
            'reported_accuracy': h['results']['scores'][0]['metrics']['accuracy']['value'],
            'recs': recs}


def single(run: dict, N: int | None = None) -> dict:
    ys = [r['y'] for r in run['recs'].values()]
    x, n = sum(ys), len(ys)
    cp, wi = clopper_pearson(x, n), wilson(x, n)
    out = {'x': x, 'n': n, 'p': x / n, 'cp95': cp, 'wilson95': wi,
           'timeouts': sum(r['timeout'] for r in run['recs'].values()),
           'noanswer_other': sum(r['noanswer_other'] for r in run['recs'].values())}
    assert abs(out['p'] - run['reported_accuracy']) < 1e-9, (run['file'], out['p'], run['reported_accuracy'])
    if N:
        out['fpc_hypergeom95_deterministic_only'] = hypergeometric_interval(x, n, N)
    return out


def paired(runX: dict, runY: dict, only=None) -> dict:
    """X = candidate (quantized), Y = reference. D = p_X - p_Y."""
    ids = sorted(runX['recs'])
    assert sorted(runY['recs']) == ids, 'id sets differ'
    for i in ids:
        assert runX['recs'][i]['item'] == runY['recs'][i]['item'], f'item {i} differs'
    if only is not None:
        ids = [i for i in ids if only(runX['recs'][i], runY['recs'][i])]
    a = b = c = d = 0
    for i in ids:
        x, y = runX['recs'][i]['y'], runY['recs'][i]['y']
        a += x and y
        b += x and not y
        c += (not x) and y
        d += (not x) and (not y)
    n = a + b + c + d
    D, lo, hi = newcombe_paired(a, b, c, d)
    return {'n': n, 'a_both': a, 'b_X_only': b, 'c_Y_only': c, 'd_neither': d,
            'pX': (a + b) / n, 'pY': (a + c) / n, 'diff_X_minus_Y': D,
            'newcombe10_95': (lo, hi), 'mcnemar_exact_p': mcnemar_exact(b, c),
            'discordance': (b + c) / n}


def paired_indicator(runX: dict, runY: dict, key: str) -> dict:
    ids = sorted(runX['recs'])
    b = sum(1 for i in ids if runX['recs'][i][key] and not runY['recs'][i][key])
    c = sum(1 for i in ids if runY['recs'][i][key] and not runX['recs'][i][key])
    return {'X_only': b, 'Y_only': c, 'mcnemar_exact_p_ASSUMES_INDEPENDENCE': mcnemar_exact(b, c)}


def fmt_ci(ci) -> str:
    return f'[{ci[0]:.3f}, {ci[1]:.3f}]'


def main() -> None:
    checks = ['see tests/python/test_stats_statlib.py']
    if len(sys.argv) < 3:
        sys.exit(__doc__)  # usage
    log_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    runs = {cfg: {m: load_run(log_dir, pre) for m, pre in tasks.items()}
            for cfg, tasks in MANIFEST.items()}
    for cfg, tasks in runs.items():
        for m, r in tasks.items():
            assert r['task'] == TASK_OF[m], (cfg, m, r['task'])

    R: dict = {'selftests': checks, 'single': {}, 'paired': {}, 'families': {},
               'power': {}, 'gpqa_decomposition': {}, 'extra': {}}

    # ---- single proportions
    for cfg, tasks in runs.items():
        for m, r in tasks.items():
            R['single'][f'{cfg}/{m}'] = dict(single(r, 198 if m == 'gpqa' else None),
                                             file=r['file'], frame=FRAME[m])

    # ---- families of paired comparisons (X = quantized/candidate, Y = reference)
    fam_defs = {
        'A_qwen_gpqa_q8_128K_vs_f16_64K': [
            ('gpqa run1', 'qwen_q8_128K', 'qwen_f16_64K', 'gpqa'),
            ('gpqa run2', 'qwen_q8_128K_rep2', 'qwen_f16_64K', 'gpqa')],
        'B_qwen_short_chain_q8_128K_vs_f16_64K': [
            (m, 'qwen_q8_128K', 'qwen_f16_64K', m)
            for m in ('gsm8k', 'humaneval', 'humaneval_plus', 'mmlu_pro', 'tools')],
        'C_gemma_all_q8_256K_vs_f16_128K': [
            (m, 'gemma_q8_256K', 'gemma_f16_128K', m)
            for m in ('gsm8k', 'humaneval', 'humaneval_plus', 'mmlu_pro', 'gpqa', 'tools')],
        'C2_gemma_same_ctx_q8_256K_vs_f16_256K': [
            (m, 'gemma_q8_256K', 'gemma_f16_256K', m) for m in ('gsm8k', 'humaneval', 'tools')],
        'D_gptoss_vllm_fp8_vs_auto': [
            (m, 'gptoss_fp8', 'gptoss_auto', m)
            for m in ('gsm8k', 'humaneval', 'humaneval_plus', 'mmlu_pro', 'gpqa', 'tools')],
    }
    # Test-retest (same configuration) references: pure decoding + scheduling noise.
    noise_defs = [
        ('qwen gpqa q8_128K run2 vs run1', 'qwen_q8_128K_rep2', 'qwen_q8_128K', 'gpqa'),
        ('gptoss humaneval_plus auto run2 vs run1', 'gptoss_auto_rep2', 'gptoss_auto', 'humaneval_plus'),
        ('gptoss tools fp8 run2 vs run1', 'gptoss_fp8_rep2', 'gptoss_fp8', 'tools'),
        ('qwen gsm8k f16 64K vs f16 256K(partial CPU)', 'qwen_f16_64K', 'qwen_f16_256K_partialCPU', 'gsm8k'),
        ('qwen humaneval f16 64K vs f16 256K(partial CPU)', 'qwen_f16_64K', 'qwen_f16_256K_partialCPU', 'humaneval'),
        ('gemma gsm8k f16 128K vs f16 256K', 'gemma_f16_128K', 'gemma_f16_256K', 'gsm8k'),
        ('gemma humaneval f16 128K vs f16 256K', 'gemma_f16_128K', 'gemma_f16_256K', 'humaneval'),
    ]
    for fam, comps in fam_defs.items():
        raw = {}
        for label, X, Y, m in comps:
            res = paired(runs[X][m], runs[Y][m])
            res.update(X=X, Y=Y, metric=m, frame=FRAME[m])
            # non-inferiority check against the owner's 0.02 acceptance margin
            res['NI_margin_0.02_lower_bound_above_-0.02'] = res['newcombe10_95'][0] > -0.02
            R['paired'][f'{fam}/{label}'] = res
            raw[label] = res['mcnemar_exact_p']
        adj = holm(raw)
        R['families'][fam] = {k: {'raw_p': raw[k], 'holm_p': adj[k]} for k in raw}
    for label, X, Y, m in noise_defs:
        res = paired(runs[X][m], runs[Y][m])
        res.update(X=X, Y=Y, metric=m)
        R['paired'][f'NOISE/{label}'] = res
    res = paired(runs['qwen_q8_128K_rep2']['gpqa'], runs['qwen_q8_128K']['gpqa'],
                 only=lambda rx, ry: not rx['timeout'] and not ry['timeout'])
    R['paired']['NOISE/qwen gpqa q8_128K run2 vs run1, completed in both'] = res

    # ---- generation length, items completed in both runs (descriptive + sign test)
    for label, X, Y, m in [('qwen gpqa run1 q8 vs f16', 'qwen_q8_128K', 'qwen_f16_64K', 'gpqa'),
                           ('qwen gpqa run2 q8 vs f16', 'qwen_q8_128K_rep2', 'qwen_f16_64K', 'gpqa'),
                           ('qwen gpqa q8 run2 vs run1', 'qwen_q8_128K_rep2', 'qwen_q8_128K', 'gpqa'),
                           ('qwen mmlu_pro q8 vs f16', 'qwen_q8_128K', 'qwen_f16_64K', 'mmlu_pro'),
                           ('qwen gsm8k q8 vs f16', 'qwen_q8_128K', 'qwen_f16_64K', 'gsm8k'),
                           ('gemma gpqa q8 vs f16', 'gemma_q8_256K', 'gemma_f16_128K', 'gpqa'),
                           ('gemma mmlu_pro q8 vs f16', 'gemma_q8_256K', 'gemma_f16_128K', 'mmlu_pro')]:
        rx, ry = runs[X][m]['recs'], runs[Y][m]['recs']
        ratios = sorted(rx[i]['tokens'] / ry[i]['tokens'] for i in rx
                        if not rx[i]['timeout'] and not ry[i]['timeout']
                        and rx[i]['tokens'] and ry[i]['tokens'])
        up = sum(1 for r in ratios if r > 1)
        dn = sum(1 for r in ratios if r < 1)
        R['extra'][f'tokens/{label}'] = {
            'n_items': len(ratios), 'median_ratio_X_over_Y': ratios[len(ratios) // 2] if ratios else None,
            'X_longer': up, 'Y_longer': dn, 'sign_test_p': mcnemar_exact(up, dn),
            'max_tokens_X': max((rx[i]['tokens'] or 0) for i in rx),
            'max_tokens_Y': max((ry[i]['tokens'] or 0) for i in ry)}

    # ---- GPQA decomposition (qwen and gemma): correct / wrong-answered / timeout
    for cfg in ('qwen_f16_64K', 'qwen_q8_128K', 'qwen_q8_128K_rep2', 'gemma_f16_128K', 'gemma_q8_256K', 'gptoss_fp8', 'gptoss_auto'):
        r = runs[cfg]['gpqa']['recs'].values()
        done = [x for x in r if not x['timeout']]
        xd = sum(x['y'] for x in done)
        R['gpqa_decomposition'][cfg] = {
            'n': len(r), 'correct': sum(x['y'] for x in r),
            'wrong_answered': sum(1 for x in done if not x['y'] and not x['noanswer_other']),
            'wrong_no_answer_other': sum(1 for x in done if x['noanswer_other']),
            'timeout_scored_wrong': sum(1 for x in r if x['timeout']),
            'acc_among_completed': (xd, len(done), xd / len(done) if done else None,
                                    clopper_pearson(xd, len(done)) if done else None)}

    # ---- sensitivity: items completed (no timeout) in BOTH runs
    for label, X, Y in [('qwen gpqa run1', 'qwen_q8_128K', 'qwen_f16_64K'),
                        ('qwen gpqa run2', 'qwen_q8_128K_rep2', 'qwen_f16_64K'),
                        ('gemma gpqa', 'gemma_q8_256K', 'gemma_f16_128K')]:
        res = paired(runs[X]['gpqa'], runs[Y]['gpqa'], only=lambda rx, ry: not rx['timeout'] and not ry['timeout'])
        R['paired'][f'SENS_completed_in_both/{label}'] = res
        R['extra'][f'timeout_pairs/{label}'] = paired_indicator(runs[X]['gpqa'], runs[Y]['gpqa'], 'timeout')

    # ---- combined qwen GPQA: f16 vs mean of the two q8 runs, exact sign-flip test
    ids = sorted(runs['qwen_f16_64K']['gpqa']['recs'])
    d2 = [(runs['qwen_q8_128K']['gpqa']['recs'][i]['y'] + runs['qwen_q8_128K_rep2']['gpqa']['recs'][i]['y'])
          - 2 * runs['qwen_f16_64K']['gpqa']['recs'][i]['y'] for i in ids]
    R['extra']['qwen_gpqa_f16_vs_mean_q8_signflip'] = {
        'mean_diff_q8_minus_f16': sum(d2) / (2 * len(d2)),
        'nonzero_items': sum(1 for v in d2 if v), 'exact_p': sign_flip_exact(d2)}

    # ---- power / MDE at the observed n and discordance
    psi_retest = R['paired']['NOISE/qwen gpqa q8_128K run2 vs run1']['discordance']
    psi_obs1 = R['paired']['A_qwen_gpqa_q8_128K_vs_f16_64K/gpqa run1']['discordance']
    psi_clean = R['paired']['NOISE/qwen gpqa q8_128K run2 vs run1, completed in both']['discordance']
    for label, n, psi in [('gpqa n=60, psi = q8 test-retest discordance', 60, psi_retest),
                          ('gpqa n=60, psi = f16-vs-q8 run1 discordance', 60, psi_obs1),
                          ('gpqa n=198 (all items, 1 run each), psi = q8 test-retest', 198, psi_retest),
                          ('gpqa n=198 (all items, 1 run each), psi = test-retest among completed', 198, psi_clean),
                          ('mmlu_pro n=100, psi=0.17 (gpt-oss fp8-vs-auto observed)', 100, 0.17)]:
        R['power'][label] = {'n': n, 'psi': psi, 'mde_80pct_exact_mcnemar': mcnemar_mde(n, psi)}
    R['power']['power_to_detect_obs_delta_0.10_n60'] = {
        'psi': psi_obs1, 'power': mcnemar_power(60, (psi_obs1 + 0.10) / 2, (psi_obs1 - 0.10) / 2)}
    plan = {}
    for psi in (0.10, 0.15, 0.20, round(psi_retest, 4)):
        n_ni = n_paired_noninferiority(psi, 0.02)
        plan[f'psi={psi}'] = {
            'n_item_answers_NI_margin_0.02': n_ni,
            'runs_per_config_on_all_198_items': ceil(n_ni / 198),
            'n_detect_true_drop_0.10': n_paired_superiority(psi, 0.10),
            'n_detect_true_drop_0.05': n_paired_superiority(psi, 0.05)}
    R['power']['follow_up_plan'] = plan

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(R, f, indent=1, default=list)
    write_md(R, os.path.join(out_dir, 'results.md'))
    print(open(os.path.join(out_dir, 'results.md')).read())


def write_md(R: dict, path: str) -> None:
    L = ['# KV-cache quantization re-analysis: machine-generated tables', '',
         'Generated by kvquant_stats.py. Intervals are 95% two-sided.', '',
         '## Self-tests', ''] + [f'- ok: {c}' for c in R['selftests']] + ['', '## Single proportions', '',
         '| run/metric | x/n | p | Clopper-Pearson | Wilson | FPC (deterministic decoding only) | timeouts |',
         '|---|---|---|---|---|---|---|']
    for k, v in R['single'].items():
        fpc = fmt_ci(v['fpc_hypergeom95_deterministic_only']) if 'fpc_hypergeom95_deterministic_only' in v else '-'
        L.append(f"| {k} | {v['x']}/{v['n']} | {v['p']:.3f} | {fmt_ci(v['cp95'])} | {fmt_ci(v['wilson95'])} | {fpc} | {v['timeouts']} |")
    L += ['', '## Paired comparisons (X = candidate, Y = reference; diff = pX - pY)', '',
          '| comparison | n | pX | pY | diff | Newcombe-10 95% CI | b (X only) | c (Y only) | McNemar p | Holm p | CI lower > -0.02 |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for k, v in R['paired'].items():
        fam, lab = k.split('/', 1)
        hp = R['families'].get(fam, {}).get(lab, {}).get('holm_p')
        hps = f'{hp:.3f}' if hp is not None else '-'
        ni = v.get('NI_margin_0.02_lower_bound_above_-0.02', '-')
        L.append(f"| {k} | {v['n']} | {v['pX']:.3f} | {v['pY']:.3f} | {v['diff_X_minus_Y']:+.3f} | "
                 f"{fmt_ci(v['newcombe10_95'])} | {v['b_X_only']} | {v['c_Y_only']} | {v['mcnemar_exact_p']:.3f} | {hps} | {ni} |")
    L += ['', '## GPQA outcome decomposition', '',
          '| config | n | correct | wrong (answered) | wrong (no answer, other) | timeout (scored wrong) | acc among completed [CP] |',
          '|---|---|---|---|---|---|---|']
    for k, v in R['gpqa_decomposition'].items():
        xd, nd, pd, ci = v['acc_among_completed']
        L.append(f"| {k} | {v['n']} | {v['correct']} | {v['wrong_answered']} | {v['wrong_no_answer_other']} | "
                 f"{v['timeout_scored_wrong']} | {xd}/{nd} = {pd:.3f} {fmt_ci(ci)} |")
    L += ['', '## Extra', '', '```', json.dumps(R['extra'], indent=1, default=list), '```',
          '', '## Power', '', '```', json.dumps(R['power'], indent=1, default=list), '```', '']
    with open(path, 'w') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    main()
