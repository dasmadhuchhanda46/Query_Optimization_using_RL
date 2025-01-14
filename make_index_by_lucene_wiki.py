# 
import pandas as pd

import os
import lucene
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory
import org.apache.lucene.document as document
from org.apache.lucene.analysis.en import EnglishAnalyzer
from java.io import File

file_path = "/home/madhu/lucene_code/Data/Wiki_movie_plots_deduped.csv"

lucene.initVM()

# Ensure index directory exists and is writable
index_dir_path = "/home/madhu/lucene_code/index"
if not os.path.exists(index_dir_path):
    os.makedirs(index_dir_path)

indexPath = File(index_dir_path).toPath()
indexDir = FSDirectory.open(indexPath)
writerConfig = IndexWriterConfig(EnglishAnalyzer())
writer = IndexWriter(indexDir, writerConfig)


def indexSinglePlot(title, plot):
    doc = document.Document()
    doc.add(document.Field('TITLE', title, document.TextField.TYPE_STORED))
    doc.add(document.Field('PLOT', plot, document.TextField.TYPE_STORED))
    writer.addDocument(doc)


def make_inverted_index(file_path):
    df = pd.read_csv(file_path)
    docid = 0
    for i in df.index:
        print(docid, "_", df['Title'][i])
        indexSinglePlot(df['Title'][i], df['Plot'][i])
        #inverted_index = {}
        docid += 1

def closeWriter():
    writer.close()

make_inverted_index(file_path)
closeWriter()