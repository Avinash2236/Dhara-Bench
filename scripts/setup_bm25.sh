#!/usr/bin/env bash
set -euo pipefail
BM25_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BM25_SETUP_PYTHON="${BM25_SETUP_PYTHON:-python3.12}"
"$BM25_SETUP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
if [[ ! -x "$BM25_BASE/.venv-bm25/bin/python" ]]; then
  "$BM25_SETUP_PYTHON" -m venv "$BM25_BASE/.venv-bm25"
fi
# Install the locked lexical-only dependency closure. Neural modules are unused.
"$BM25_BASE/.venv-bm25/bin/python" -m pip install --no-deps -r "$BM25_BASE/scripts/requirements.txt"
"$BM25_BASE/.venv-bm25/bin/python" - "$BM25_BASE" <<'PY'
import hashlib,json,pathlib,sys,tarfile,urllib.request
base=pathlib.Path(sys.argv[1]); runtime=base/'.runtime'; runtime.mkdir(exist_ok=True)
config=json.loads((base/'scripts/java_runtime.json').read_text())
archive=runtime/config['name']
if not archive.exists(): urllib.request.urlretrieve(config['download_url'],archive)
with archive.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
assert actual==config['archive_sha256'], 'Java archive checksum mismatch'
with tarfile.open(archive) as stream:
    root=stream.getnames()[0].split('/')[0]
    if not (runtime/root/'bin/java').exists(): stream.extractall(runtime,filter='data')
link=runtime/'java21'
if not link.exists(): link.symlink_to(root,target_is_directory=True)
assert link.resolve()==(runtime/root).resolve(), 'Existing Java link differs from pinned runtime'
(runtime/'java_download.json').write_text(json.dumps(config,indent=2)+'\n')
PY
source "$BM25_BASE/scripts/env_bm25.sh"
"$JAVA_HOME/bin/java" -version
"$BM25_PYTHON" -c "from pyserini.pyclass import autoclass; print('Lucene', autoclass('org.apache.lucene.util.Version').LATEST.toString())"
