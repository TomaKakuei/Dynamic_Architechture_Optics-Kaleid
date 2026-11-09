"""Check a text-only source snapshot without importing or running experiments."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    allowed={'.py','.json','.txt','.csv'}
    published=set()
    python_count=0
    for row in manifest['files']:
        path=(ROOT/row['path']).resolve()
        assert path.is_relative_to(ROOT/'workspace'),row['path']
        assert path.suffix in allowed,row['path']
        assert path not in published,row['path']
        published.add(path)
        raw=path.read_bytes()
        assert len(raw)==row['bytes'],row['path']
        assert hashlib.sha256(raw).hexdigest()==row['published_sha256'],row['path']
        if path.suffix=='.py':
            ast.parse(raw.decode('utf-8-sig'),filename=row['path'],feature_version=(3,11))
            assert row['source_sha256']==row['published_sha256'],row['path']
            python_count+=1
        elif path.suffix=='.json':json.loads(raw.decode('utf-8-sig'))
    actual={p.resolve() for p in (ROOT/'workspace').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual==published,'Unlisted or missing workspace files'
    print(json.dumps({'status':'passed','text_files':len(published),
                      'unchanged_python_sources':python_count,'binary_files':0,
                      'bytes':manifest['bytes']},indent=2))

if __name__=='__main__':main()
