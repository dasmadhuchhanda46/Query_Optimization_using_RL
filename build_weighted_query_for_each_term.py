from lucene import initVM, JavaError
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.store import FSDirectory
from java.io import StringReader, File
from collections import Counter
import os

# Initialize the Lucene VM
initVM()

# Path to your index and relevance file
index_path = "/home/madhu/lucene_code/index_trec"
relevance_file = "/home/madhu/lucene_code/trec_data/lucene_data/trec678rb/qrels/robust_601-700.qrel"

# Open the index
directory = FSDirectory.open(File(index_path).toPath())
reader = DirectoryReader.open(directory)

# Analyzer
analyzer = EnglishAnalyzer()

# Read relevance data file
current_query_id = None
term_counter = Counter()  # Initialize term counter
results = {}  # Store results for each query ID

with open(relevance_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:  # Ensure proper format
            query_id = parts[0]  # Extract the query ID
            doc_id = parts[2]  # Extract the document ID
            relevance = int(parts[3])  # Extract the relevance score

            # Check if the query ID changes
            if query_id != current_query_id:
                if current_query_id is not None:
                    # Store results for the previous query ID
                    most_frequent_terms = term_counter.most_common(10)
                    print("--------------------------")
                    print(current_query_id)
                    print("--------------------------")
                    print(most_frequent_terms)
                    results[current_query_id] = most_frequent_terms

                # Reset term counter for the new query ID
                current_query_id = query_id
                term_counter = Counter()

            # Process only relevant documents
            if relevance > 0:
                # Fetch document from the index
                doc = None
                for i in range(reader.maxDoc()):
                    document = reader.document(i)
                    if document.get("DOCNO") == doc_id:  # Adjust field name as per your schema
                        doc = document
                        break

                if doc:
                    content = doc.get("TEXT")  # Adjust field name as per your index schema

                    # Tokenize and count terms
                    token_stream = analyzer.tokenStream("", StringReader(content))
                    try:
                        token_stream.reset()
                        while token_stream.incrementToken():
                            term = token_stream.getAttribute(CharTermAttribute.class_).toString()
                            term_counter[term] += 1
                    finally:
                        token_stream.close()

# Store the results for the last query ID
if current_query_id is not None:
    most_frequent_terms = term_counter.most_common(10)
    results[current_query_id] = most_frequent_terms

# Print results for each query ID
for query_id, terms in results.items():
    print(f"Query ID: {query_id}")
    print(f"Top 10 frequent terms: {terms}")
    print()

# Close resources
reader.close()
directory.close()
