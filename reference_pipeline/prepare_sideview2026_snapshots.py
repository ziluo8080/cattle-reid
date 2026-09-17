import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(r'E:\research\cattle_reid\data\raw\sideview2026'); OUT=ROOT/'processed-snapshots-fullbody-letterbox-v1'; T=224; PAD=(128,128,128)
def h(b): return hashlib.sha256(b).hexdigest()
OUT.mkdir(parents=True,exist_ok=True); rows=[]; arr=[]
with zipfile.ZipFile(ROOT/'snapshots.zip') as z:
 names=sorted(n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg','.png')) and n.startswith('snapshots/images/'))
 for i,n in enumerate(names):
  raw=z.read(n)
  with Image.open(io.BytesIO(raw)) as im:
   im=ImageOps.exif_transpose(im).convert('RGB'); ow,oh=im.size; s=min(T/ow,T/oh); nw,nh=max(1,round(ow*s)),max(1,round(oh*s)); im=im.resize((nw,nh),Image.Resampling.LANCZOS); c=Image.new('RGB',(T,T),PAD); left,top=(T-nw)//2,(T-nh)//2; c.paste(im,(left,top)); arr.append(np.asarray(c,dtype=np.uint8).transpose(2,0,1).copy())
  rel=Path(n).relative_to('snapshots/images'); rows.append({'index':i,'scene':'snapshots','identity':rel.parts[0],'sample_key':rel.stem,'relative_path':str(rel).replace('\\','/'),'source_sha256':h(raw),'original_width':ow,'original_height':oh,'resized_width':nw,'resized_height':nh,'pad_left':left,'pad_top':top})
np.save(OUT/'images_uint8.npy',np.stack(arr),allow_pickle=False); (OUT/'manifest.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8')
p={'name':'sideview2026-snapshots-fullbody-letterbox-v1','target_size':[224,224],'color':'RGB','exif_transpose':True,'resize':'aspect-preserving LANCZOS','padding':{'mode':'constant','rgb':list(PAD)},'crop':False,'normalization':'ImageNet mean/std at inference','image_count':len(rows),'identity_count':len({x["identity"] for x in rows})}; (OUT/'protocol.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); (OUT/'processing_receipt.json').write_text(json.dumps({'protocol':p,'array_sha256':h((OUT/'images_uint8.npy').read_bytes()),'manifest_sha256':h((OUT/'manifest.jsonl').read_bytes())},indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps(p,ensure_ascii=False))
