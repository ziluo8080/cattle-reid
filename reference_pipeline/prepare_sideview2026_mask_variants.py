"""基于公开 mask 生成固定规则输入对照；不改变模型和身份协议。"""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(r'E:/research/cattle_reid/data/raw/sideview2026'); ZIP=ROOT/'snapshots.zip'; OUT=ROOT/'processed-snapshots-mask-variants-v1'; T=224

def sha(b): return hashlib.sha256(b).hexdigest()
def letter(im):
    im=ImageOps.exif_transpose(im).convert('RGB'); ow,oh=im.size; s=min(T/ow,T/oh); nw,nh=max(1,round(ow*s)),max(1,round(oh*s)); x=im.resize((nw,nh),Image.Resampling.LANCZOS); c=Image.new('RGB',(T,T),(128,128,128)); c.paste(x,((T-nw)//2,(T-nh)//2)); return np.asarray(c,dtype=np.uint8).transpose(2,0,1).copy(),[ow,oh]
def masked(im,mask,mode):
    im=ImageOps.exif_transpose(im).convert('RGB'); mask=ImageOps.exif_transpose(mask).convert('L').resize(im.size,Image.Resampling.NEAREST); a=np.asarray(mask)>0
    if mode=='neutral':
        x=np.asarray(im).copy(); x[~a]=128; return Image.fromarray(x)
    ys,xs=np.where(a)
    if len(xs)==0: raise ValueError('empty mask')
    margin=max(4,round(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)*.08)); l=max(0,xs.min()-margin); r=min(im.width,xs.max()+1+margin); t=max(0,ys.min()-margin); b=min(im.height,ys.max()+1+margin)
    return im.crop((l,t,r,b))
def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=[]; variants={v:[] for v in ('bbox8pct','neutral128')}
    with zipfile.ZipFile(ZIP) as z:
      names=sorted(n for n in z.namelist() if n.startswith('snapshots/images/') and n.lower().endswith('.jpg'))
      for i,n in enumerate(names):
        raw=z.read(n); mn=n.replace('/images/','/masks/').rsplit('.',1)[0]+'.png';
        with Image.open(io.BytesIO(raw)) as im, Image.open(io.BytesIO(z.read(mn))) as mk:
          for v in variants:
            out,shape=letter(masked(im,mk,'neutral' if v=='neutral128' else 'bbox'))
            variants[v].append(out)
        rel=Path(n).relative_to('snapshots/images'); rows.append({'index':i,'scene':'snapshots','identity':rel.parts[0],'sample_key':rel.stem,'relative_path':str(rel).replace('\\','/'),'source_sha256':sha(raw)})
    for v,arr in variants.items():
      d=OUT/v; d.mkdir(parents=True,exist_ok=True); np.save(d/'images_uint8.npy',np.stack(arr),allow_pickle=False); (d/'manifest.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8'); (d/'protocol.json').write_text(json.dumps({'name':f'sideview2026-{v}-v1','base_protocol':'sideview2026-snapshots-fullbody-letterbox-v1','mask_source':'snapshots/masks','target_size':[224,224],'rule':'mask bbox + 8% margin' if v=='bbox8pct' else 'mask outside filled RGB(128,128,128)','training':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'images':len(rows),'identities':len({r['identity'] for r in rows}),'variants':list(variants)},ensure_ascii=False))
if __name__=='__main__': main()
