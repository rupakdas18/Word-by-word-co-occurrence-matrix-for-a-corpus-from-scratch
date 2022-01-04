# Word-by-word-co-occurrence-matrix-for-a-corpus-from-scratch

**Program Details:**
This program creates a word by word co-occurrence matrix for a corpus. The cell values in this matrix
should be the Pointwise Mutual Information (PMI) between 2 words.

**Code Usage:**
The user must provide below input in below format:
similiar-pmi.py 6 ./PA4-News-2011 input.txt
where,
similiar-pmi.py = Program file name
6 = window size
./PA4-News-2011 = location of training corpus
input.txt = contains pair words
the output format will be:
cosine(word1,word2) word1 word2 count(word1) count(word2) count(word1,word2) PMI(word1,word2)

**Program Algorithm:**
1) Read data file
  - loop through each file and read each line
  - remove all non-alphanumeric characters and convert all text to lower case.
2) Create the word by word co-occurrence matrix
  - create word pairs with a given window size and store them in a dictionary with frequency count.
  - create a n*n dataframe where n is the number of unique words. This is a word by word co-occurrence matrix where the cell value is frequency.
  - Create another dataframe where cell values are PMI values (using PMI equation)
3) Get the result
  - Read the input file line by line
  - for each pair of words, find the PMI value from the dataframe. Convert to numpy array, find the dot product and finally the cosine value.
  - Display the output
