# Targeted Sentiment Analysis

## Description
We have gathered a new data set from students at the University of Michigan and developed a new method for targeted sentiment. We perform both entity extraction and sentiment analysis over extracted entities showing improvements over previous work on a similar task.

Using natural language processing and machine learning techniques we build a pipeline which consists of an entity extraction system, formulated as a sequence labeling task, which feeds into a sentiment analysis classifier which takes a target entity and surrounding text as input and labels expressed sentiment toward the entity as positive, negative, or neutral. The development of new domain specific features for both parts of the pipeline lead us to improvements over several baselines.

## Supporting New Entity Types

If you want to reuse the code for new types of entities you can now define the entity types you wish to support in the settings.py file. You can define entity types using the ENT_TYPES dictionary which has a key for the name of each type. The annotated data must have '<type' lines to identify them. See the following example:

```Python
# Kathe Halverson was the only aspect of EECS 555 Parallel Computing that I liked
# <instructor
# name=Kathe Halverson
# sentiment=positive>
# <class
# id=555
# name=Parallel Computing
# sentiment=negative>

ENT_TYPES = {'instructor': ['name'], \
             'class': ['name', 'department', 'id'] \
            }
```

The rest of the code will then use these types. The list of identifiers in this dictionary are used to mark token types for the CRF tagger. When merging these into entities it combines identifiers belonging to the same entity type as long as it does not see another identifier of the same type. For example, if you have an utterance that lists '492, the AI class and the data structures class', it will merge the id '492' and the name 'AI' into one class but when it sees the second name, 'data structures' it creates a second entity.

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

The Python wrapper we are using for Stanford CoreNLP only works with Python 2.

### How to run the pipeline
1. Run make_crfbioyz.py to generate the tokenized file to train the CRFs.
2. Run splits.py to generate splits for the data. You can change the number of splits in the file -- default is 100.
3. Run crf_tagger.py to generate taggers. This will read in the 'crf-input-data' file as well as the splits file and generate an individual tagger for each split.
4. Run sentiment_class.py which will read in the EECS_annotated_samples file and use the generated splits to create a different sentiment classifier for each split.
5. Run nlu.py to use the whole pipeline for each split. This will push each utterance through the splits tagger and then use the output as input for the sentiment classifier for that split.

### Where files are output
1. Output for make_crfbioyz.py is stored in 'crf-input-data'.
2. Output for splits.py is stored in 'splits'.
3. Output for crf_tagger.py is stored in /taggers - pscores, rscores, and fscores.
4. Output for sentiment_class.py is stored in /classifiers - sentiment_scores.
5. Output for nlu.py is stored in nlu_scores in an array with the format of total precision and recall, followed by the precision and recall for each entity type.

### Comments about already installed dependencies
These steps are required to set up dependencies.

To install pywrapper we ran:
`git clone https://github.com/brendano/stanford_corenlp_pywrapper dependencies/`
`ln -s dependencies/stanford_corenlp_pywrapper .`

Extracted Stanford CoreNLP into dependencies folder. The code has been tested with versions 2015-04-20 and 2018-02-27.

The dataset can be downloaded from http://web.eecs.umich.edu/~mihalcea/downloads/targetedSentiment.2017.tar.gz and should be placed in this folder.

These lexicons are in the data folder:
1. [Bing Liu's lexicon](https://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)
2. [MPQA subjectivity lexicon](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/)

Renamed Bing Liu's lexicon files into `neg_words` and `pos_words`.

Removed 'm' characters from line 5549 and 5550 in MPQA file.

## Publication

If you use this code please cite:

```
@inproceedings{Welch16Targeted,
    author = {Welch, C. and Mihalcea, R.},
    title = {Targeted Sentiment to Understand Student Comments},
    booktitle = {Proceedings of the International Conference on Computational Linguistics (COLING 2016)},
    address = {Osaka, Japan},
    year = {2016}
}
```
