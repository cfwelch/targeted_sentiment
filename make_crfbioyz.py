

import sys

from settings import *
from stanford_corenlp_pywrapper import CoreNLP

with open('EECS_annotated_samples_anonymized') as handle:
    lines = handle.readlines()
    lines = [line.strip() for line in lines]

proc = CoreNLP('pos', corenlp_jars=[PATH_TO_STANFORD_CORENLP])

out_file = open('crf-input-data', 'w')
cur_line, cur_parsed, cur_mapped, cur_pos = [None]*4
current_nonos = 0
in_annotations = False
in_type = None

for line in lines:
    if line == '':
        current_nonos += sum([1 for tok in cur_mapped if tok != 'O'])
        #print('Non-Os is now: ' + str(current_nonos))
        #print(cur_mapped)
        #print('\n\n\n')
        assert len(cur_mapped) == len(cur_pos) and len(cur_pos) == len(cur_parsed)
        for i in range(0, len(cur_mapped)):
            out_file.write(cur_parsed[i] + '\t' + cur_pos[i] + '\t' + cur_mapped[i] + '\n')
        out_file.write('\n')
        in_annotations = False
        continue
    if not in_annotations:
        cur_line = line.replace('/', ' / ').replace('EECS', ' EECS ').replace('eecs', ' eecs ').replace('  ', ' ')
        cur_parsed = proc.parse_doc(cur_line)
        cur_pttok, cur_postk = [], []
        for sent in cur_parsed['sentences']:
            cur_pttok.extend(sent['tokens'])
            cur_postk.extend(sent['pos'])
        cur_parsed = cur_pttok
        cur_pos = cur_postk
        #print(line)
        #print(cur_parsed)
        cur_mapped = ['O'] * len(cur_pttok)
        current_nonos = 0
        in_annotations = True
    else:
        # parse annotations
        anno = line[1:] if line.startswith('<') else line
        anno = anno[:-1] if line.endswith('>') else anno
        #print('annotation is: ' + str(anno))
        aparts = anno.split('=')
        if len(aparts) == 1:
            in_type = aparts[0]

        if in_type not in ENT_TYPES:
            print('Entity type found in the data is not in the defined set of entities in your settings.py file.')
            sys.exit(1)
        assert len(aparts) < 3

        if aparts[0] in ALL_IDS:
            id_type = aparts[0]
            match_parts = aparts[1].split(' ')
            for i in range(0, len(cur_parsed)-len(match_parts)+1):
                this_match = True
                for j in range(0, len(match_parts)):
                    if cur_parsed[i+j] != match_parts[j]:
                        this_match = False
                        break

                if this_match:
                    for j in range(0, len(match_parts)):
                        cur_mapped[i+j] = in_type + '_' + id_type + '_' + ('B' if j == 0 else 'I')
        # else:
        #     print('Warning: Annotation part \'' + aparts[0] + '\' is not in the set of IDs from your settings.py file.')
