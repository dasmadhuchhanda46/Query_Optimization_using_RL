import lucene, os
import xml.etree.ElementTree as ET
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader, MultiTerms
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
# from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from java.io import StringReader, File
from collections import Counter

# Initialize Lucene VM
lucene.initVM()

# Paths
index_path = "/home/madhu/lucene_code/index_trec"  # Path to Lucene index
query_file = "/home/madhu/lucene_code/trec_data/lucene_data/trec678rb/topics/robust.xml"  # Path to XML query file
output_file = "query_terms.txt"  # Output file to save the terms

try:
    os.remove(output_file)
except:
    pass  # Ignore if the file does not exist

# Open Lucene index
index_directory = FSDirectory.open(File(index_path).toPath())
index_reader = DirectoryReader.open(index_directory)
index_searcher = IndexSearcher(index_reader)
analyzer = EnglishAnalyzer()

# Parse the XML file
queries = ET.parse(query_file).getroot()

# Open the output file
with open(output_file, 'a') as out_file:
    # Process each query
    for query_element in queries:
        query_num = query_element[0].text  # Query number
        query_text = query_element[1].text  # Query text

        print(f"Processing Query {query_num}: {query_text}")

        # Parse and execute the query
        query = QueryParser("TEXT", analyzer).parse(query_text)
        top_docs = index_searcher.search(query, 10).scoreDocs  # Retrieve top 10 documents

        # Dictionary to store term frequencies
        term_frequencies = Counter()

        # Process each document in the top 10
        for hit in top_docs:
            doc_id = hit.doc  # Document ID
            doc = index_searcher.doc(doc_id)
            text = doc.get("TEXT")
            if text:
                stream = analyzer.tokenStream("", StringReader(text))
                try:
                    stream.reset()
                    while stream.incrementToken():
                        current_word = stream.getAttribute(CharTermAttribute.class_).toString()
                        term_frequencies[current_word] += 1
                finally:
                    stream.close()
            # term_vector = index_reader.getTermVector(doc_id, "TEXT")  # Retrieve term vector for the "TEXT" field
            
            # if term_vector is None:
            #     print(f"Warning: No term vector for doc_id {doc_id}. Skipping...")
            #     continue
            
            # terms_enum = term_vector.iterator()  # Initialize the iterator
            # while terms_enum.next() is not None:
            #     term = terms_enum.term().utf8ToString()  # Get term text
            #     freq = terms_enum.totalTermFreq()  # Get term frequency
            #     if freq > 0:  # Ensure frequency is positive
            #         term_frequencies[term] += freq  # Update frequency in the counter


        # Retrieve the top 100 terms
        top_100_terms = term_frequencies.most_common(100)
        print(top_100_terms)

        # Write the terms to the output file
        out_file.write(f"Query {query_num}: {query_text}\n")
        for rank, (term, freq) in enumerate(top_100_terms, start=1):
            out_file.write(f"{rank}. Term: {term}, Frequency: {freq}\n")
        out_file.write("\n")

print(f"Top terms for all queries have been saved in '{output_file}'.")
