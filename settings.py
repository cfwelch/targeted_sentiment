

PATH_TO_STANFORD_CORENLP = '/local/cfwelch/stanford-corenlp-full-2018-02-27/*'

NUM_SPLITS = 1
TRAIN_SIZE = 0.67 # two thirds of the total data

# Define entity types here
# Each type must have a list of identifiers that do not contain the '_' token

# Example of an annotation:
# 
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

# Shortcut for checking ids
ALL_IDS = list(set([i for j in ENT_TYPES for i in ENT_TYPES[j]]))
