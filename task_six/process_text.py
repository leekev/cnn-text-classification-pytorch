# 1. Preprocess text using torchtext

# 1. Preprocess text
# ^^^^^^^^^^^^^^^^^^^^^^^

import pandas as pd
import numpy as np
import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
# from torchtext.vocab import Vectors, GloVe


def latin2utf(fn):
	with open(fn, 'r', encoding='latin1') as f:
		lines = f.readlines()

	ll = [l.encode('utf-8') for l in lines]

	with open(fn.split('.csv')[0] + '.u.csv', 'w', encoding='utf-8') as f:
		for i, l in enumerate(ll):
			if (i == 0) :
				f.write("id" + "," + l.decode())
			else :
				f.write(str(i) + "," + l.decode())

latin2utf("./train/train_hillary.csv")
latin2utf("./train/val_hillary.csv")
latin2utf("./test/test_hillary.csv")

# ## Declare Fields
# tokenize = lambda x: x.split()
# TEXT = Field(sequential=True, tokenize=tokenize, lower=True, vectors=GloVe(name='6B', dim=300))
 
# LABEL = Field(sequential=False, use_vocab=True)

# ## Construct Dataset
# datafields = [("Tweet", TEXT),
# 			  ("Target", None),
#               ("Stance", LABEL),
#               ("Opinion Towards", None),
#               ("Sentiment", None)]

# trn, vld = TabularDataset.splits(
# 	               path="./",
# 	               train='train.utf8.csv', validation='val.utf8.csv',
# 	               format='csv',
# 	               skip_header=True,
# 	               fields=datafields)

# tst = TabularDataset(
#            path="./test.csv",
#            format='csv',
#            skip_header=True,
#            fields=datafields)


# TEXT.build_vocab(trn)
# LABEL.build_vocab(trn)

# ## Build Iterator
# from torchtext.data import Iterator, BucketIterator
 
# train_iter, val_iter = BucketIterator.splits(
#  (trn, vld), # we pass in the datasets we want the iterator to draw data from
#  batch_sizes=(64, 64),
#  sort_key=lambda x: len(x.Tweet),
#  sort_within_batch=False,
#  repeat=False
# )

# test_iter = BucketIterator(tst, batch_size=64, sort=False, sort_within_batch=False, repeat=False)






