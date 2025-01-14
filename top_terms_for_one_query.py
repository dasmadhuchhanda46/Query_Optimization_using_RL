from lucene import initVM, JavaError
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from java.io import StringReader, File
from collections import Counter

# Initialize Lucene VM
initVM()

# Path to the Lucene index
index_path = "/home/madhu/lucene_code/index_trec"
query_text = "Turkey Iraq water"
field = "TEXT"  # Field to search in the index

# Open the Lucene index
index_directory = FSDirectory.open(File(index_path).toPath())
index_reader = DirectoryReader.open(index_directory)
index_searcher = IndexSearcher(index_reader)
analyzer = EnglishAnalyzer()

# Parse and execute the query
query = QueryParser(field, analyzer).parse(query_text)
top_docs = index_searcher.search(query, 10).scoreDocs  # Retrieve top 10 documents

# Dictionary to store term frequencies
term_frequencies = Counter()

print(top_docs)
# Process each document in the top 10
for hit in top_docs:
    doc_id = hit.doc  # Document ID of Lucene
    doc = index_searcher.doc(doc_id)
    # print(doc)
    text = doc.get("TEXT")
    if text:  # Ensure the document's text field is not null
        stream = analyzer.tokenStream("", StringReader(text))
        try:
            stream.reset()
            while stream.incrementToken():
                current_word = stream.getAttribute(CharTermAttribute.class_).toString()
                term_frequencies[current_word] += 1
        finally:
            stream.close()  # Ensure the TokenStream is properly closed

# Retrieve the top 100 terms
top_100_terms = term_frequencies.most_common(100)

# Print the results
print(f"Top 100 terms from the top 10 documents for query '{query_text}':")
for rank, (term, freq) in enumerate(top_100_terms, start=1):
    print(f"{rank}. Term: {term}, Frequency: {freq}")
