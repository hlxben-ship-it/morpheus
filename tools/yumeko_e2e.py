"""Drive one whole Yumeko job through the running server's API, as the page does. Long: renders about 60 pictures and trains."""
import json, sys, time, urllib.request
B = 'http://127.0.0.1:5196/api/yumeko/'
AGENT = sys.argv[1] if len(sys.argv) > 1 else 'naggie'
def post(a, **k):
    try:
        return json.load(urllib.request.urlopen(urllib.request.Request(B + a, json.dumps(k).encode(), {'Content-Type': 'application/json'}), timeout=60))
    except urllib.error.HTTPError as e:
        raise SystemExit(f'{a}: {e.read().decode()[:400]}')
def get(p): return json.load(urllib.request.urlopen(B + p, timeout=60))
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def idle(i, t=1500):
    end = time.time() + t
    while time.time() < end:
        j = get('job/' + i)
        if not j['busy']:
            if j['error']: raise SystemExit('job error: ' + j['error'])
            return j
        time.sleep(3)
    raise SystemExit('timed out')
j = post('create', agent=AGENT); i = j['id']; log('created', i, j['agent']['trigger_token'], repr(j['agent']['identity_summary'][:90]))
post('previews', id=i); j = idle(i); log('styles', j['style_previews'])
post('style', id=i, style='modern')
post('chat', id=i, text='Design her as an anime character from her agent description, keep her signature colours, standing, simple background. Render a grid.')
j = idle(i); log('chat', j['phase1']['chat'][-1]['text'][:120], len(j['phase1']['grid']), j['phase1']['tags'][:3])
if not j['phase1']['grid']:
    post('chat', id=i, text='Render the grid now.'); j = idle(i); log('chat2', len(j['phase1']['grid']))
post('approve', id=i, file=j['phase1']['grid'][0]['file']); j = get('job/' + i); log('phase', j['phase'], 'identity', [t['tag'] for t in j['bases'][0]['tags'] if t['role'] == 'identity'])
post('phase2', id=i, positive=j['phase2']['positive'] + ', from side', negative=j['phase2']['negative']); j = idle(i); log('p2 grid', len(j['phase2']['grid']))
post('approve', id=i, file=j['phase2']['grid'][0]['file']); j = get('job/' + i); log('phase', j['phase'])
while len(j['dataset']['accepted']) + len(j['dataset']['batch']) < 30:
    post('batch', id=i); j = idle(i)
    for c in j['dataset']['batch']:
        post('decide', id=i, file=c['file'], keep=True)
    j = get('job/' + i); log('round', j['dataset']['rounds'], 'planned', [c['planned'][0] for c in j['dataset']['batch']])
post('train', id=i); log('training')
while True:
    j = get('job/' + i); t = j['train']
    if t['state'] != 'running': break
    time.sleep(30)
log('train', t['state'], t.get('step'), t.get('stages'), (t.get('log_tail') or '')[-300:])
if t['state'] != 'done': raise SystemExit(1)
post('stage_test', id=i); j = idle(i); log('stages', j['stage_tests'])
post('stage', id=i, stage=2000)
post('use', id=i, prompt='close-up, smile, garden, daytime'); j = idle(i); log('use', j['gallery'][0]['file'])
log('E2E OK', i)
