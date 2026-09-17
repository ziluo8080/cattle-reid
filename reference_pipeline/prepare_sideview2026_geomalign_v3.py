"""只用 mask 定位主体外接框，保留真实像素，避免分割误差制造伪前景。"""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(r'E:/research/cattle_reid/data/raw/sideview2026'); ZIP=ROOT/'snapshots.zip'; OUT=ROOT/'processed-snapshots-geomalign-v3'; T=224; MAXSIDE=200; MARGIN=.15
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=[]; arr=[]
 with zipfile.ZipFile(ZIP) as z:
  names=sorted(n for n in z.namelist() if n.startswith('snapshots/images/') and n.lower().endswith('.jpg'))
  for i,n in enumerate(names):
   raw=z.read(n); mn=n.replace('/images/','/masks/').rsplit('.',1)[0]+'.png'
   with Image.open(io.BytesIO(raw)) as im,Image.open(io.BytesIO(z.read(mn))) as mk:
    im=ImageOps.exif_transpose(im).convert('RGB'); mk=ImageOps.exif_transpose(mk).convert('L').resize(im.size,Image.Resampling.NEAREST); ys,xs=np.where(np.asarray(mk)>0)
    if not len(xs): raise ValueError(n)
    w,h=im.size; bw,bh=xs.max()-xs.min()+1,ys.max()-ys.min()+1; margin=max(8,round(max(bw,bh)*MARGIN)); l=max(0,xs.min()-margin); r=min(w,xs.max()+margin+1); t=max(0,ys.min()-margin); b=min(h,ys.max()+margin+1); crop=im.crop((l,t,r,b)); cw,ch=crop.size; scale=min(MAXSIDE/cw,MAXSIDE/ch); nw,nh=max(1,round(cw*scale)),max(1,round(ch*scale)); crop=crop.resize((nw,nh),Image.Resampling.LANCZOS); canvas=Image.new('RGB',(T,T),(128,128,128)); left,top=(T-nw)//2,(T-nh)//2; canvas.paste(crop,(left,top)); arr.append(np.asarray(canvas,dtype=np.uint8).transpose(2,0,1).copy())
   rel=Path(n).relative_to('snapshots/images'); rows.append({'index':i,'scene':'snapshots','identity':rel.parts[0],'sample_key':rel.stem,'relative_path':str(rel).replace('\\','/'),'source_sha256':sha(raw),'bbox':[int(l),int(t),int(r),int(b)],'bbox_margin_fraction':MARGIN,'resized_size':[nw,nh],'pad':[left,top]})
 np.save(OUT/'images_uint8.npy',np.stack(arr),allow_pickle=False); (OUT/'manifest.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8'); p={'name':'sideview2026-geomalign-v3','base_protocol':'sideview2026-snapshots-image-holdout-54id-v1','mask_use':'bbox localization only; original pixels retained','bbox_margin':MARGIN,'body_max_side':MAXSIDE,'target_size':[224,224],'padding_rgb':[128,128,128],'training':False,'model_changed':False,'image_count':len(rows),'identity_count':len({r["identity"] for r in rows})}; (OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n'); (OUT/'processing_receipt.json').write_text(json.dumps({'protocol':p,'array_sha256':sha((OUT/'images_uint8.npy').read_bytes()),'manifest_sha256':sha((OUT/'manifest.jsonl').read_bytes())},ensure_ascii=False,indent=2)+'\n'); print(json.dumps(p,ensure_ascii=False))
if __name__=='__main__': main()
