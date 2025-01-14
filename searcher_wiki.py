import lucene
import org.apache.lucene.document as document
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher
from java.io import File


def search(index_path, q):
    print("Searching for:", q)
    lucene.initVM()
    analyzer = EnglishAnalyzer()
    directory = FSDirectory.open(File(index_path).toPath())
    searcher = IndexSearcher(DirectoryReader.open(directory))
    searcher.setSimilarity(BM25Similarity(1.2, 0.75))
    query = QueryParser("PLOT", analyzer).parse(q)
    scoreDocs = searcher.search(query, 10).scoreDocs

    print(f"{len(scoreDocs)} total matching documents")
    for scoreDoc in scoreDocs:
        doc = searcher.doc(scoreDoc.doc)
        print(doc.get("TITLE"),":", doc.get("PLOT"))


search("/home/madhu/lucene_code/index", "crime and Murder")