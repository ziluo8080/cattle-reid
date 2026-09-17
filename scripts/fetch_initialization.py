"""从PyTorch官方地址获取历史ImageNet初态，完整SHA256不符则停止。"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/assets/densenet121-a639ec97.pth')
    args = parser.parse_args()
    config = json.loads((ROOT/'protocols/training_config.json').read_text())
    if not args.output.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.download')
        if temporary.exists():
            raise FileExistsError('Previous partial download exists; inspect before removing it')
        urllib.request.urlretrieve('https://download.pytorch.org/models/densenet121-a639ec97.pth', temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != config['pretrained_sha256']:
            raise ValueError('Downloaded initialization checksum mismatch')
        temporary.replace(args.output)
    if hashlib.sha256(args.output.read_bytes()).hexdigest() != config['pretrained_sha256']:
        raise ValueError('Existing initialization checksum mismatch')
    print('PASS:', args.output)


if __name__ == '__main__':
    main()
