# E1-246-Natural-Language-Understanding-2019--Assignment1
This repository contains codes, report, word embeddings for assignment1

Files A1_w4.py and A1_w5.py produce the word embeddings of dimension 300 sampling 10 negative context words and 
half window size 4 and 5 respectively. These both give superior results in Word Similarity Task and Word Anological Task.

These embeddings have been uploaded to google drive https://drive.google.com/drive/folders/1nN0vDfkjhC6n7R8NJ-kMR2GnTSFhE9It?usp=sharing with access to sawankumar@iisc.ac.in

To run the code, type "python A1_w4.py" or "python A1_w5.py" in command line

Word Similarity Task can be done by ipython notebook "Simlex Scores.ipynb" while sequentially running the snippets

Word Analogical Task can be done by ipython notebook "Analogical Reasoning Task.ipynb" while sequentially running the snippets

For both tasks, first we need to create word embeddings as word embeddings are saved in .npy format and can not be uploaded
on the github due to large size.

For reproducbility of results word embedding of size 128 and 256 have been uploaded.
