from lucene import initVM, JavaError
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.util import BytesRef
from java.io import StringReader, File
from collections import Counter
import os

initVM()
# Path to your index and relevance file
index_path = "/home/madhu/lucene_code/index_trec"
relevance_file = "/home/madhu/lucene_code/test.txt"

# Open the index
directory = FSDirectory.open(File(index_path).toPath())
reader = DirectoryReader.open(directory)

# Analyzer
analyzer = EnglishAnalyzer()

# Initialize term frequency counter
term_counter = Counter()

# Read relevance data file
with open(relevance_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:  # Ensure proper format
            doc_id = parts[2]
            relevance = int(parts[3])

            # Process only relevant documents
            if relevance > 0:
                # Fetch document from index
                doc = None
                for i in range(reader.maxDoc()):
                    document = reader.document(i)
                    # print(document.get("DOCNO"))
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


# Get most frequent terms
most_frequent_terms = term_counter.most_common(10)  # Adjust number of terms

# Print results
print("Most frequent terms:", most_frequent_terms)

# Close resources
reader.close()
directory.close()
