"""只去除原始图像四边的连续黑色边框，保留牛体和真实背景。"""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(r'E:/research/cattle_reid/data/raw/sideview2026'); ZIP=ROOT/'snapshots.zip'; OUT=ROOT/'processed-snapshots-blacktrim-v1'; T=224; THRESH=8; FRACTION=.95; MAX_TRIM=.4
def sha(b): return hashlib.sha256(b).hexdigest()
def trim(im):
 a=np.asarray(im); h,w=a.shape[:2]; l,t,r,b=0,0,w,h
 black=(a.max(2)<=THRESH); row=black.mean(1)>=FRACTION; col=black.mean(0)>=FRACTION
 while t+2 < b and t/h < MAX_TRIM and row[t]: t+=1
 while b-2 > t and (h-b)/h < MAX_TRIM and row[b-1]: b-=1
 while l+2 < r and l/w < MAX_TRIM and col[l]: l+=1
 while r-2 > l and (w-r)/w < MAX_TRIM and col[r-1]: r-=1
 return im.crop((l,t,r,b)),[l,t,r,b]
def letter(im):
 w,h=im.size; s=min(T/w,T/h); nw,nh=max(1,round(w*s)),max(1,round(h*s)); x=im.resize((nw,nh),Image.Resampling.LANCZOS); c=Image.new('RGB',(T,T),(128,128,128)); c.paste(x,((T-nw)//2,(T-nh)//2)); return np.asarray(c,dtype=np.uint8).transpose(2,0,1).copy(),[nw,nh]
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=[]; vals=[]
 with zipfile.ZipFile(ZIP) as z:
  names=sorted(n for n in z.namelist() if n.startswith('snapshots/images/') and n.lower().endswith('.jpg'))
  for i,n in enumerate(names):
   raw=z.read(n)
   with Image.open(io.BytesIO(raw)) as image: im=ImageOps.exif_transpose(image).convert('RGB'); crop,bbox=trim(im); value,size=letter(crop); vals.append(value)
   rel=Path(n).relative_to('snapshots/images'); rows.append({'index':i,'scene':'snapshots','identity':rel.parts[0],'sample_key':rel.stem,'relative_path':str(rel).replace('\\','/'),'source_sha256':sha(raw),'trim_bbox':bbox,'resized_size':size})
 np.save(OUT/'images_uint8.npy',np.stack(vals),allow_pickle=False); (OUT/'manifest.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8'); p={'name':'sideview2026-blacktrim-v1','base_protocol':'sideview2026-snapshots-image-holdout-54id-v1','rule':'remove only continuous edge rows/columns with >=95% RGB max<=8, cap 40% per edge','target_size':[224,224],'resize':'aspect-preserving LANCZOS','padding_rgb':[128,128,128],'mask_used':False,'training':False,'model_changed':False,'image_count':len(rows),'identity_count':len({r["identity"] for r in rows})}; (OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); (OUT/'processing_receipt.json').write_text(json.dumps({'protocol':p,'array_sha256':sha((OUT/'images_uint8.npy').read_bytes()),'manifest_sha256':sha((OUT/'manifest.jsonl').read_bytes())},ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'images':len(rows),'trimmed':sum(r['trim_bbox']!=[0,0,1170,1170] for r in rows),'protocol':p},ensure_ascii=False))
if __name__=='__main__': main()
