"""使用GitHub官方CLI下载私有附件，逐文件校验；不处理或保存认证令牌。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/assets')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    catalog = json.loads((ROOT/'protocols/release_assets.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for item in catalog['assets']:
        path = args.output/item['name']
        if not path.exists() and not args.verify_only:
            subprocess.run(['gh', 'release', 'download', catalog['tag'], '--repo', catalog['repository'],
                            '--pattern', item['name'], '--dir', str(args.output)], check=True)
        if not path.exists() or path.stat().st_size != item['size'] or sha256(path) != item['sha256']:
            raise ValueError('Missing/corrupt asset: '+str(path)+'; no automatic overwrite')
        print('Verified', item['name'], flush=True)
        if args.extract and item.get('extract'):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    target = (args.output/info.filename).resolve()
                    if not target.is_relative_to(args.output.resolve()) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Unsafe ZIP member')
                    if not info.is_dir() and target.exists():
                        if target.read_bytes() != archive.read(info):
                            raise FileExistsError('Refusing to replace differing file: '+str(target))
                archive.extractall(args.output)
    print('PASS: all release assets verified')


if __name__ == '__main__':
    main()
