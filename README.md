# Targeted Sentiment Analysis

## Description
We have gathered a new data set from students at the University of Michigan and developed a new method for targeted sentiment. We perform both entity extraction and sentiment analysis over extracted entities showing improvements over previous work on a similar task.

Using natural language processing and machine learning techniques we build a pipeline which consists of an entity extraction system, formulated as a sequence labeling task, which feeds into a sentiment analysis classifier which takes a target entity and surrounding text as input and labels expressed sentiment toward the entity as positive, negative, or neutral. The development of new domain specific features for both parts of the pipeline lead us to improvements over several baselines.

## Instructions

### Dependencies

The following packages need to be installed:
```
pip install python-crfsuite nltk sklearn scipy numpy pyparsing
```
Furthermore, the project needs Java 8. One way to install it is:
```
sudo add-apt-repository ppa:webupd8team/java
sudo apt update; sudo apt install oracle-java8-installer oracle-java8-set-default
```

### How to run the pipeline
1. Run splits.py to generate splits for the data. You can change the number of splits in the file -- default is 100.
2. Run crf_example.py to generate taggers. This will read in the CRFNERADVISE-BIOYZ file as well as the splits file and generate an individual tagger for each split.
3. Run sentiment_class.py which will read in the EECS_annotated_samples file and use the generated splits to create a different sentiment classifier for each split.
4. Run NLU.py to use the whole pipeline for each split. This will push each utterance through the splits tagger and then use the output as input for the sentiment classifier for that split.

### Where files are output
1. Output for splits.py is stored in 'splits'.
2. Output for crf_example.py is stored in /taggers - pscores, rscores, and fscores
3. Output for sentiment_class.py is stored in /classifiers - sentiment_scores
4. Output for NLU.py is stored in NLU_scores

### Comments about already installed dependencies
These steps are required to set up dependencies.

To install pywrapper we did:
`git clone https://github.com/brendano/stanford_corenlp_pywrapper dependencies/`
`ln -s dependencies/stanford_corenlp_pywrapper .`

Extracted http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip into dependencies folder.

The dataset can be downloaded from http://web.eecs.umich.edu/~mihalcea/downloads/targetedSentiment.2017.tar.gz and should be placed in this folder.

These lexicons are in the data folder:
1. [Bing Liu's lexicon](https://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)
2. [MPQA subjectivity lexicon](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/)

Renamed Bing Liu's lexicon files into `neg_words` and `pos_words`.
Removed 'm' characters from line 5549 and 5550 in MPQA file.

Created directories `taggers` and `classifiers`.

## Publication

If you use this code please cite:

```
@inproceedings{Welch16Targeted,
    author = {Welch, C. and R. Mihalcea},
    title = {Targeted Sentiment to Understand Student Comments},
    booktitle = {Proceedings of the International Conference on Computational Linguistics (COLING 2016)},
    address = {Japan},
    year = {2016}
}
```
