#!/usr/bin/env python3
"""Extract per-sample records from inspect_ai .eval logs (read-only).
Usage: python3 kvquant_extract_samples.py <out.json> <log1.eval> [<log2.eval> ...]
Per sample: id, item hash (sha256 of input + choices + target), score value
(C/I or numeric), answer, target, output tokens, times, error, limit, stop
reason, reasoning chars, answer chars."""
import hashlib, json, sys, zipfile, os

def item_hash(s):
    inp = s.get('input')
    if not isinstance(inp, str):
        inp = json.dumps(inp, sort_keys=True)
    blob = json.dumps([inp, s.get('choices'), s.get('target')], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

def score_of(s):
    sc = s.get('scores') or {}
    if not sc:
        return None, None
    name, v = next(iter(sc.items()))
    return v.get('value'), v.get('answer')

def out_info(s):
    out = s.get('output') or {}
    ch = (out.get('choices') or [{}])[0]
    msg = ch.get('message') or {}
    content = msg.get('content')
    rchars = achars = 0
    if isinstance(content, list):
        for c in content:
            if c.get('type') == 'reasoning':
                rchars += len(c.get('reasoning') or '')
            elif c.get('type') == 'text':
                achars += len(c.get('text') or '')
    elif isinstance(content, str):
        achars = len(content)
    return ch.get('stop_reason'), rchars, achars, (out.get('usage') or {})

def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)  # usage
    out_path, logs = sys.argv[1], sys.argv[2:]
    res = {}
    for p in logs:
        z = zipfile.ZipFile(p)
        h = json.loads(z.read('header.json'))
        recs = []
        for nm in sorted(n for n in z.namelist() if n.startswith('samples/')):
            s = json.loads(z.read(nm))
            val, ans = score_of(s)
            stop, rch, ach, usage = out_info(s)
            recs.append({
                'id': s.get('id'), 'epoch': s.get('epoch'), 'item': item_hash(s),
                'target': s.get('target') if isinstance(s.get('target'), str) else json.dumps(s.get('target'))[:40],
                'value': val, 'answer': ans,
                'output_tokens': usage.get('output_tokens'),
                'total_time': s.get('total_time'), 'working_time': s.get('working_time'),
                'error': (s.get('error') or {}).get('message') if s.get('error') else None,
                'limit': s.get('limit'), 'stop_reason': stop,
                'reasoning_chars': rch, 'answer_chars': ach,
                'error_retries': len(s.get('error_retries') or []),
                'started_at': s.get('started_at'), 'completed_at': s.get('completed_at'),
            })
        recs.sort(key=lambda r: (str(type(r['id'])), r['id']))
        ev = h['eval']
        res[os.path.basename(p)] = {
            'task': ev.get('task'), 'model': ev.get('model'), 'created': ev.get('created'),
            'completed_at': (h.get('stats') or {}).get('completed_at'),
            'n': len(recs), 'samples': recs,
            'accuracy': ((h.get('results') or {}).get('scores') or [{}])[0].get('metrics', {}).get('accuracy', {}).get('value'),
            'max_connections': (ev.get('config') or {}).get('max_connections'),
            'time_limit': (ev.get('config') or {}).get('time_limit'),
            'generate_config': ev.get('model_generate_config'),
        }
    json.dump(res, open(out_path, 'w'), indent=1)

if __name__ == '__main__':
    main()
