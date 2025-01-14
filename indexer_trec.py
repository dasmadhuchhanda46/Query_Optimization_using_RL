import os
import sys
from bs4 import BeautifulSoup
import lucene
from java.io import File
# from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.analysis.en import EnglishAnalyzer
import org.apache.lucene.document as document

def init_lucene(index_dir):
    """Initializes the Lucene index writer."""
    lucene.initVM()
    indexPath = File(index_dir).toPath()
    indexDir = FSDirectory.open(indexPath)
    writerConfig = IndexWriterConfig(EnglishAnalyzer())
    return IndexWriter(indexDir, writerConfig)

def indexDoc(writer, docno, title, text):
    """Indexes a single document."""
    doc = document.Document()
    doc.add(document.Field("DOCNO", docno, document.TextField.TYPE_STORED))
    doc.add(document.Field("TITLE", title, document.TextField.TYPE_STORED))
    doc.add(document.Field("TEXT", text, document.TextField.TYPE_STORED))
    writer.addDocument(doc)

def parse_and_index(writer, filename, prefix, title_tag):
    """Parses and indexes documents based on the prefix and title tag."""
    global doccount
    with open(filename, 'r', encoding="ISO-8859-1") as fp:
        soup = BeautifulSoup(fp, 'html.parser')
        doc = soup.find("doc")
        while doc is not None:
            docno = doc.findChildren("docno")[0].get_text().strip()
            title = doc.findChildren(title_tag)
            text = doc.findChildren("text")
            if len(text) == 0:
                doc = doc.find_next("doc")
                continue
            text = text[0].get_text().strip()
            title = "" if len(title) == 0 else title[0].get_text().strip()
            print(f'{doccount} -- {docno} -> {title}')
            indexDoc(writer, docno, title, text)
            doc = doc.find_next("doc")
            doccount += 1

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 indexer.py /path/to/trec-robust-collection/ /path/to/directory/where/index/should/be/stored")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    writer = init_lucene(output_dir)

    global doccount
    doccount = 0
    for filename in os.listdir(input_dir):
        full_path = os.path.join(input_dir, filename)
        if os.path.isfile(full_path):
            try:
                if filename.startswith('fb'):
                    parse_and_index(writer, full_path, 'fb', 'ti')
                elif filename.startswith('ft'):
                    parse_and_index(writer, full_path, 'ft', 'headline')
                elif filename.startswith('la'):
                    parse_and_index(writer, full_path, 'la', 'headline')
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    writer.close()

if __name__ == "__main__":
    main()
