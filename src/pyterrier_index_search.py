from time import time
import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict

import pyterrier as pt
pt.init()

from jnius import autoclass
tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

def terrier_tokenizer(raw_utterance):
    new_utterance = " ".join(tokeniser.getTokens(raw_utterance))
    return new_utterance


def get_indexes_and_lexicon():
    index_ref = pt.IndexRef.of("/data3/muntean/conversational-cache/indexes/CAST2020-stemmed/data.properties")
    index = pt.IndexFactory.of(index_ref)

    di = index.getDirectIndex()
    doi = index.getDocumentIndex()
    lex = index.getLexicon()

    print(index.getCollectionStatistics())
    return index, di, doi, lex


