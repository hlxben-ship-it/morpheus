"""Finish a trained job: test stages, keep the last, render one. Usage: yumeko_finish.py <job id>"""
import json, sys, time, urllib.request
B='http://127.0.0.1:5196/api/yumeko/'; i=sys.argv[1]
def post(a,**k): return json.load(urllib.request.urlopen(urllib.request.Request(B+a,json.dumps(k).encode(),{'Content-Type':'application/json'}),timeout=60))
def get(p): return json.load(urllib.request.urlopen(B+p,timeout=60))
def idle():
    while True:
        j=get('job/'+i)
        if not j['busy']:
            if j['error']: raise SystemExit(j['error'])
            return j
        time.sleep(3)
j=get('job/'+i); print(j['phase'],j['train']['state'],j['train'].get('step'),flush=True)
post('stage_test',id=i); j=idle(); print(j['stage_tests'],flush=True)
post('stage',id=i,stage=2000); post('use',id=i,prompt='close-up, smile, garden, daytime'); j=idle(); print(j['gallery'][0]['file'],'OK')
