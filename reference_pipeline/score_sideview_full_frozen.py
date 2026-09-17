import json,sys,hashlib,time
from pathlib import Path
import numpy as np, torch
BASE=Path('/q1/cattle-reid/cows2021-threeway-v1'); R=Path('/localdisk-tmp/sideview2026-input'); O=Path('/q1/cattle-reid/sideview2026-frozen-external-snapshots-v1'); O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(BASE/'src')); from cattle_reid_gate0.strong_rgb_densenet import DenseRGB
rows=[json.loads(x) for x in (R/'manifest.jsonl').read_text().splitlines()]; arr=np.load(R/'images_uint8.npy',mmap_mode='r'); proto=json.loads((R/'protocol_index.json').read_text()); models=json.loads((BASE/'locks/cows2021-fullcandidate-threeway-v1/models.json').read_text()); dev='cuda' if torch.cuda.is_available() else 'cpu'; mean=torch.tensor([.485,.456,.406],device=dev)[None,:,None,None]; std=torch.tensor([.229,.224,.225],device=dev)[None,:,None,None]; byid={}
for i,r in enumerate(rows): byid.setdefault(r['identity'],[]).append(i)
results=[]; started=time.time()
for mi,spec in enumerate(models,1):
 m=DenseRGB().eval().to(dev); m.load_state_dict(torch.load(spec['path'],map_location='cpu',weights_only=True),strict=True); em=[]
 with torch.inference_mode():
  for i in range(0,len(arr),64):
   x=(torch.from_numpy(np.asarray(arr[i:i+64])).float().to(dev)/255-mean)/std; z=torch.nn.functional.normalize(m(x),dim=1); assert torch.isfinite(z).all(); em.append(z.cpu().numpy())
 emb=np.concatenate(em)
 for draw in proto['draws']:
  d=draw['by_identity']; ids=proto['candidate_ids']
  for k in (1,3,5):
   correct=total=0; per={}
   gallery={j:emb[d[j]['support'][str(k)]].mean(0) for j in ids}
   for ident in ids:
    c=0; q=d[ident]['query']
    for qi in q:
     pred=max(ids,key=lambda j:float(gallery[j]@emb[qi])); c+=int(pred==ident)
    per[ident]={'correct':c,'total':len(q)}; correct+=c; total+=len(q)
   results.append({'model':f"{spec['method']}-fold{spec['fold']}-seed{spec['seed']}",'method':spec['method'],'fold':spec['fold'],'seed':spec['seed'],'draw':draw['draw'],'k':k,'correct':correct,'total':total,'accuracy':correct/total,'per_identity':per})
 print(f'{mi}/{len(models)} {spec["method"]}-fold{spec["fold"]}-seed{spec["seed"]}',flush=True); del m
payload={'protocol_sha256':hashlib.sha256((R/'protocol_index.json').read_bytes()).hexdigest(),'preprocessing':'sideview2026-snapshots-fullbody-letterbox-v1','models':len(models),'results':results,'training':False,'device':dev,'elapsed_seconds':time.time()-started}
(O/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)); print('DONE',len(results),flush=True)
