"""Private local cache, namespaced by input metadata and implementation."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def cache_directory():
    from games import asset_paths
    root=Path(__file__).resolve().parents[1]
    assets={}
    for source in ('tactical','broadcast'):
        for name,value in asset_paths(source=source).items():
            if name=='video':continue
            p=Path(value).resolve()
            assets[f'{source}/{name}']=([str(p),p.stat().st_size,p.stat().st_mtime_ns]
                if p.is_file() else [str(p),'not installed'])
    digest=hashlib.sha256(json.dumps(assets,sort_keys=True).encode())
    files=list((root/'possession').rglob('*.py'))+list((root/'possession').rglob('*.json'))
    files+=list((root/'starter').glob('*.py'))+[root/'games.py']
    for p in sorted(files):
        digest.update(str(p.relative_to(root)).encode());digest.update(p.read_bytes())
    base=Path(os.environ.get('POSSESSION_CACHE_DIR',str(root/'.cache'))).resolve()
    result=base/digest.hexdigest()[:20]
    result.mkdir(parents=True,exist_ok=True)
    return result
