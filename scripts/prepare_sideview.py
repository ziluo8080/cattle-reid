"""按历史四种规则重建输入；绑定原图顺序和哈希，不重新选择身份或划分。"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def transform(image, mask, variant):
    """保留历史插值、round取整、外接框边界和灰色填充语义。"""
    image = ImageOps.exif_transpose(image).convert('RGB')
    max_side = 224
    if variant in ('neutral128', 'geomalign_v3'):
        mask = ImageOps.exif_transpose(mask).convert('L').resize(image.size, Image.Resampling.NEAREST)
        foreground = np.asarray(mask) > 0
        if variant == 'neutral128':
            pixels = np.asarray(image).copy()
            pixels[~foreground] = 128
            image = Image.fromarray(pixels)
        else:
            ys, xs = np.where(foreground)
            if not len(xs):
                raise ValueError('empty mask')
            margin = max(8, round(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)*.15))
            image = image.crop((max(0, xs.min()-margin), max(0, ys.min()-margin),
                                min(image.width, xs.max()+margin+1), min(image.height, ys.max()+margin+1)))
            max_side = 200
    elif variant == 'blacktrim_v1':
        a = np.asarray(image)
        h, w = a.shape[:2]
        l, t, r, b = 0, 0, w, h
        black = a.max(2) <= 8
        row, col = black.mean(1) >= .95, black.mean(0) >= .95
        while t+2 < b and t/h < .4 and row[t]:
            t += 1
        while b-2 > t and (h-b)/h < .4 and row[b-1]:
            b -= 1
        while l+2 < r and l/w < .4 and col[l]:
            l += 1
        while r-2 > l and (w-r)/w < .4 and col[r-1]:
            r -= 1
        image = image.crop((l, t, r, b))
    elif variant != 'baseline':
        raise ValueError('unknown variant')
    scale = min(max_side/image.width, max_side/image.height)
    nw, nh = max(1, round(image.width*scale)), max(1, round(image.height*scale))
    canvas = Image.new('RGB', (224, 224), (128, 128, 128))
    canvas.paste(image.resize((nw, nh), Image.Resampling.LANCZOS), ((224-nw)//2, (224-nh)//2))
    return np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'neutral128', 'geomalign_v3', 'blacktrim_v1'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    rows = [json.loads(s) for s in (ROOT/'protocols/sideview_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    values = []
    with zipfile.ZipFile(args.zip) as archive:
        for row in rows:
            name = 'snapshots/images/' + row['relative_path']
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != row['source_sha256']:
                raise ValueError('Original image hash mismatch: ' + name)
            with Image.open(io.BytesIO(raw)) as image:
                if args.variant in ('neutral128', 'geomalign_v3'):
                    mask_name = name.replace('/images/', '/masks/').rsplit('.', 1)[0]+'.png'
                    with Image.open(io.BytesIO(archive.read(mask_name))) as mask:
                        values.append(transform(image, mask, args.variant))
                else:
                    values.append(transform(image, None, args.variant))
    args.output.mkdir(parents=True)
    target = args.output/'images_uint8.npy'
    np.save(target, np.stack(values), allow_pickle=False)
    receipt = {'variant': args.variant, 'shape': list(np.stack(values).shape),
               'array_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'source_images_verified': len(rows), 'training': False}
    (args.output/'processing_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
