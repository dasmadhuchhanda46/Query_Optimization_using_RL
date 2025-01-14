import os
import sys
from bs4 import BeautifulSoup
import lucene
from java.io import File

from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import Term
from org.apache.lucene.search import TermQuery
from collections import Counter


# Path to your index
index_path = "/home/madhu/lucene_code/index_trec"

lucene.initVM() 

# Open the index
directory = FSDirectory.open(File(index_path).toPath())
reader = DirectoryReader.open(directory)

# Analyzer
analyzer = EnglishAnalyzer()

# Relevant document IDs
relevant_docs = ["FBIS3-12202", "FBIS3-13355", "FBIS3-13604"]

# Initialize term frequency counter
term_counter = Counter()

# Iterate through relevant documents
for doc_id in relevant_docs:
    doc = reader.document(doc_id)
    content = doc.get("TEXT")  # Adjust field name as per your index schema
    
    # Tokenize and count terms
    token_stream = analyzer.tokenStream("content", content)
    token_stream.reset()
    while token_stream.incrementToken():
        term = token_stream.getAttribute("CharTermAttribute").toString()
        term_counter[term] += 1
    token_stream.close()

# Get most frequent terms
most_frequent_terms = term_counter.most_common(10)  # Adjust number of terms

# Print results
print("Most frequent terms:", most_frequent_terms)

# Close resources
reader.close()
directory.close()
