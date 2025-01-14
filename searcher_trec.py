import xml.etree.ElementTree as ET  # Library for parsing XML files
import os, sys  # Module for file operations
import lucene  # Required for Lucene functionality
from org.apache.lucene.search.similarities import BM25Similarity  # Import BM25 Similarity
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
# from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer

from java.io import File

# Initialize Lucene VM
lucene.initVM()

# Remove the existing rank file to avoid appending to previous results
try:
    os.remove("rank_file")
except:
    pass  # Ignore if the file does not exist

if len(sys.argv) != 3:
    print("Usage: python3 searcher_trec.py /path/to/the-index-folder /path/to/query-file")
    sys.exit(1)



# Parse the query XML file (ensure the correct path to `robust.xml` is provided)
q = sys.argv[2]
topics = ET.parse(q).getroot()

# Open a file to write the ranking results
rank_file = open("rank_file", 'a')

field = "TEXT"  # The field in the documents to search (assumes "TEXT" field exists)
indx = sys.argv[1]
index_path = FSDirectory.open(File(indx).toPath())  # Path to the Lucene index
index_reader = DirectoryReader.open(index_path)  # Open the index for reading
index_searcher = IndexSearcher(index_reader)  # Create an IndexSearcher
index_searcher.setSimilarity(BM25Similarity())  # Set BM25 as the scoring similarity

analyzer = EnglishAnalyzer()  # Analyzer for parsing the query

print(f"Retrieving top 1000 ranked documents for each query using BM25--")

# Iterate over each query in the XML file
for top in topics:
    query_num = top[0].text  # Extract query number from the first child element
    title = top[1].text  # Extract the query text (title) from the second child element

    print(query_num, title)  # Print the query number and text for reference
    rank = 1  # Initialize the rank counter for this query

    # Parse the query using the specified field
    query = QueryParser(field, analyzer).parse(title)

    # Search the index for the top 10 documents matching the query
    hits = index_searcher.search(query, 1000).scoreDocs

    # Retrieve and write ranked documents
    for hit in hits:
        doc_id = hit.doc  # Document ID in the index
        doc_score = hit.score  # BM25 score
        doc_no = index_reader.document(doc_id).get("DOCNO")  # Get the document number
        print(f"Ranking docno {doc_no}", end="...")
        rank_file.write(f"{query_num} Q0 {doc_no} {rank} {doc_score} MyRun\n")  # Write results in TREC format
        print("Done...")
        rank += 1  # Increment the rank for the next document

# Close the rank file after processing all queries
rank_file.close()
