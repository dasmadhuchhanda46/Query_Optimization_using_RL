import lucene

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory
from java.io import StringReader
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer, SimpleAnalyzer, StopAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer


import pandas as pd
from collections import Counter


lucene.initVM()

test = """The quick brown fox jumps over the lazy dog. quick brown 
jump  quick"""

print("raw Data:", test)

print("EnglishAnalyzer")
analyzer = EnglishAnalyzer()
stream = analyzer.tokenStream("", StringReader(test))
stream.reset()
tokens = []
d = Counter()
while stream.incrementToken():
    current_word = stream.getAttribute(CharTermAttribute.class_).toString()
    tokens.append(current_word)
    d[current_word] += 1
print(tokens)
print(d)