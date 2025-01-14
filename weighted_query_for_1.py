import lucene
from org.apache.lucene.search import BooleanQuery, TermQuery, BooleanClause, BoostQuery
from org.apache.lucene.index import Term
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.analysis.standard import StandardAnalyzer
from java.nio.file import Paths
from org.apache.lucene.search.similarities import BM25Similarity

# Initialize Lucene
lucene.initVM()

# Open the index
index_path = "/home/madhu/lucene_code/index_trec"
directory = FSDirectory.open(Paths.get(index_path))
reader = DirectoryReader.open(directory)
searcher = IndexSearcher(reader)
searcher.setSimilarity(BM25Similarity())  # Use BM25 for scoring

# Define the weighted terms
weighted_terms = {
    "Turkey": 0.1,  # term1 has a weight of 2.0
    "Iraq": 10.0,  # term2 has a weight of 1.5
    "water": 0.01,
    "Turkish": 0.6,   # term3 has a default weight of 1.0
    "river" : 500.0
}

# Create a BooleanQuery to combine weighted terms
query_builder = BooleanQuery.Builder()

for term, weight in weighted_terms.items():
    term_query = TermQuery(Term("TEXT", term))  # Create a TermQuery for each term
    boosted_query = BoostQuery(term_query, weight)  # Wrap the term query in a BoostQuery with the specified weight
    query_builder.add(boosted_query, BooleanClause.Occur.SHOULD)  # Combine terms with SHOULD

weighted_query = query_builder.build()  # Build the final query

# Search for documents matching the weighted query
top_docs = searcher.search(weighted_query, 25)  # Retrieve top 25 documents

# Print the results
with open("output_for_wt_query.txt", 'a') as out_file:
    out_file.write(f"{weighted_terms.items()}\n")
    for score_doc in top_docs.scoreDocs:
        doc = searcher.doc(score_doc.doc)
        docno = doc.get("DOCNO")  # Replace "DOCNO" with your document's unique identifier field
        print(f"Document: {docno}, Score: {score_doc.score}")
        out_file.write(f"Document: {docno}, Score: {score_doc.score} \n")

