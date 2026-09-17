"""正式评估的不可变证据与跨平台备份读取；不加载模型或测试分数。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile


class EvaluationError(ValueError):
    """评估来源、阶段顺序或文件完整性不满足冻结合同。"""


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('ascii')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def signed(value):
    if 'canonical_sha256' in value:
        raise EvaluationError('cannot sign an already signed document')
    return {**value, 'canonical_sha256': hashlib.sha256(canonical_bytes(value)).hexdigest()}


def validate_signed(value):
    if not isinstance(value, dict) or signed({k: v for k, v in value.items()
                                             if k != 'canonical_sha256'}) != value:
        raise EvaluationError('canonical digest mismatch')
    return value


def publish_once(path, value):
    """允许相同内容恢复，拒绝覆盖已有证据及并发发布不同内容。"""
    path = Path(path)
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise EvaluationError(f'immutable artifact conflict: {path.name}')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise EvaluationError(f'immutable artifact conflict: {path.name}')
    finally:
        temp.unlink(missing_ok=True)


class BackupStore:
    """以 SHA 保存内容，用原始 Linux 路径索引，避免冒号和长路径改名。"""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.index = validate_signed(json.loads((self.root / 'index.json').read_bytes()))

    def path(self, name):
        try:
            entry = self.index['inventory']['files'][name]
        except KeyError as error:
            raise EvaluationError(f'backup member missing: {name}') from error
        checksum = entry['sha256']
        if not isinstance(checksum, str) or not re.fullmatch('[a-f0-9]{64}', checksum):
            raise EvaluationError('invalid object digest')
        path = self.root / 'objects' / checksum
        if (not path.is_file() or path.is_symlink() or path.stat().st_size != entry['bytes']
                or digest(path) != checksum):
            raise EvaluationError(f'backup member digest mismatch: {name}')
        return path

    def read(self, name):
        return json.loads(self.path(name).read_bytes())


def restore_backup(archive, inventory, expected_sha256, destination):
    """全包验签后恢复内容寻址副本；只有完整成员集合通过才发布索引。"""
    root = Path(destination).resolve()
    if digest(archive) != expected_sha256:
        raise EvaluationError('archive digest mismatch')
    (root / 'objects').mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.isfile() or member.name in seen:
                raise EvaluationError('archive contains duplicate or non-file member')
            seen.add(member.name)
            source = tar.extractfile(member)
            if member.name == 'backup_inventory.json':
                if json.load(source) != inventory:
                    raise EvaluationError('embedded inventory mismatch')
                continue
            entry = inventory['files'].get(member.name)
            if (entry is None or entry['bytes'] != member.size
                    or not re.fullmatch('[a-f0-9]{64}', entry['sha256'])):
                raise EvaluationError('archive member size or digest declaration mismatch')
            target = root / 'objects' / entry['sha256']
            checksum = hashlib.sha256()
            temp = None
            try:
                with tempfile.NamedTemporaryFile(dir=root / 'objects', delete=False) as output:
                    temp = Path(output.name)
                    for block in iter(lambda: source.read(4 * 1024**2), b''):
                        checksum.update(block)
                        output.write(block)
                if checksum.hexdigest() != entry['sha256']:
                    raise EvaluationError('archive member digest mismatch')
                if target.exists():
                    if target.is_symlink() or digest(target) != entry['sha256']:
                        raise EvaluationError('existing object digest mismatch')
                else:
                    os.link(temp, target)
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
    if seen != set(inventory['files']) | {'backup_inventory.json'}:
        raise EvaluationError('archive member set incomplete')
    publish_once(root / 'index.json', signed({'artifact_kind': 'evaluation-backup-index-v1',
                 'archive_sha256': expected_sha256, 'inventory': inventory}))
    return BackupStore(root)
