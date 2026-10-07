#!/usr/bin/env python3
import importlib.metadata
import platform
import subprocess
import zipfile
from bm25_common import *
import pyserini
from pyserini.pyclass import autoclass

queries, corpus = load_inputs()
index = Path((WORK / 'current_index.txt').read_text().strip())
# Canonical Anserini SimpleSearcher, through Pyserini's configured Java bridge.
# This is the same Java object called by pyserini.search.lucene.LuceneSearcher.
Searcher = autoclass('io.anserini.search.SimpleSearcher')
searcher = Searcher(str(index))
assert searcher.get_total_num_docs() == len(corpus)
searcher.set_bm25(float(0.9), float(0.4))
Analyzer = autoclass('io.anserini.analysis.DefaultEnglishAnalyzer')
analyzer = Analyzer.newStemmingInstance('porter')
searcher.set_analyzer(analyzer)
QueryGenerator = autoclass('io.anserini.search.query.BagOfWordsQueryGenerator')
generator = QueryGenerator()
Version = autoclass('org.apache.lucene.util.Version')
java = subprocess.run(['java', '-version'], capture_output=True, text=True, check=True)
jar = next((Path(pyserini.__file__).parent / 'resources/jars').glob('*fatjar.jar'))
with zipfile.ZipFile(jar) as archive:
    manifest = archive.read('META-INF/MANIFEST.MF').decode()
    pom_versions = {name: archive.read(name).decode() for name in archive.namelist()
                    if name.endswith('pom.properties') and ('io.anserini/' in name or 'org.apache.lucene/lucene-core/' in name)}
metadata_update(status='retrieving', java_version=(java.stdout + java.stderr).strip(),
                java_runtime_archive=json.loads((BASE / '.runtime/java_download.json').read_text()),
                lucene_version=str(Version.LATEST.toString()), anserini_jar=jar.name, anserini_jar_sha256=sha(jar),
                anserini_manifest=manifest, anserini_lucene_maven_properties=pom_versions,
                index_path=str(index), index_configuration={'collection':'JsonCollection','generator':'DefaultLuceneDocumentGenerator',
                    'threads':1,'analyzer':'io.anserini.analysis.DefaultEnglishAnalyzer', 'stemmer':'porter',
                    'stopwords':'Anserini built-in default English stop set', 'keepStopwords':False,
                    'storePositions':True,'storeDocvectors':True,'storeRaw':True,'search_field':'contents'},
                retrieval_api='pyserini.pyclass.autoclass(io.anserini.search.SimpleSearcher); canonical Java implementation used by LuceneSearcher',
                query_generator='io.anserini.search.query.BagOfWordsQueryGenerator',
                query_sanitization='None. Exact workbook string passed to canonical BagOfWordsQueryGenerator; analyzer tokenization, English stopword removal, and Porter stemming only.',
                tie_breaking='Canonical Anserini SimpleSearcher score order with document-ID tie breaking; no Python reordering.',
                package_versions={name:importlib.metadata.version(name) for name in ['pyserini','pyjnius','numpy','pandas','matplotlib','openpyxl']})
counts = []
with (OUT / 'run_bm25_top1000.trec').open('w', encoding='utf-8') as stream:
    for n, query in enumerate(queries, 1):
        hits = searcher.search(generator, query['question'], 1000)
        if not hits:
            raise RuntimeError(f'No returned documents for query {query["query_id"]}; cannot satisfy run coverage')
        docids = [int(hit.docid) for hit in hits]
        assert len(docids) == len(set(docids)) and all(docid in corpus for docid in docids)
        for rank, hit in enumerate(hits, 1):
            stream.write(f'{query["query_id"]} Q0 {hit.docid} {rank} {float(hit.score):.17g} {RUN_NAME}\n')
        counts.append({'query_id':query['query_id'],'returned_documents':len(hits)})
        if n % 25 == 0 or n == len(queries):
            print(f'Retrieved {n}/{len(queries)} queries', flush=True)
searcher.close()
write_csv(OUT / 'retrieval_counts.csv', counts, ['query_id','returned_documents'])
metadata_update(status='retrieved', run_sha256=sha(OUT / 'run_bm25_top1000.trec'),
                queries_with_fewer_than_1000_results=sum(row['returned_documents'] < 1000 for row in counts),
                short_run_policy='Return up to 1000 matching documents; do not fabricate or pad zero-score documents. Precision denominator remains k.')
