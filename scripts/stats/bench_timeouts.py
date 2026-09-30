#!/usr/bin/env python3
"""List every inspect_ai .eval log with >= 1 time-limited sample (read-only).
Usage: python3 bench_timeouts.py <inspect_log_dir> <out.tsv>"""
import json, os, sys, zipfile
if len(sys.argv) < 3:
    sys.exit(__doc__)  # usage
d, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(os.listdir(d)):
    if not f.endswith('.eval'):
        continue
    z = zipfile.ZipFile(os.path.join(d, f))
    if 'header.json' not in z.namelist():
        continue
    h = json.loads(z.read('header.json'))
    if h.get('status') != 'success':
        continue
    n = lim = 0
    for nm in z.namelist():
        if nm.startswith('samples/'):
            n += 1
            lim += bool(json.loads(z.read(nm)).get('limit'))
    if lim:
        ev = h['eval']
        acc = h['results']['scores'][0]['metrics']['accuracy']['value']
        rows.append([ev['created'][:19], ev['task'], ev['model'], ev.get('model_base_url', ''), str(lim), str(n), f'{acc:.3f}', f])
with open(out, 'w') as fh:
    fh.write('created\ttask\tmodel\tbase_url\ttime_limited\tn\treported_accuracy\tfile\n')
    for r in rows:
        fh.write('\t'.join(r) + '\n')
print(len(rows), 'logs with >= 1 time-limited sample ->', out)
