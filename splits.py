

import random
import NLU

from settings import *

######################################################################################
# To stratify with (1-TRAIN_SIZE) test split across:
#   1. Entity Types
#   2. Sentiment Classes
#   3. Token Types
#
# We record the total number of each in the data set, shuffle the set and
# then continue to add things as long as adding an utterance does not exceed any of
# the limits defined by count*(1-TRAIN_SIZE).
######################################################################################

def main():
	allines = NLU.getALines()
	allU = NLU.getUtterances(allines)
	slots = [NLU.getSlots(i) for i in allU]
	ent_counts, sent_counts, tok_counts = NLU.getSlotStats(slots)

	print('Number of Sentences: ' + str(len(allU)))
	print('Entity Counts: ' + str(ent_counts))
	print('Sentiment Counts: ' + str(sent_counts))
	print('Token Counts: ' + str(tok_counts))

	ent_lim = {k: v*(1.0-TRAIN_SIZE) for k,v in ent_counts.items()}
	sent_lim = {k: v*(1.0-TRAIN_SIZE) for k,v in sent_counts.items()}
	tok_lim = {k: v*(1.0-TRAIN_SIZE) for k,v in tok_counts.items()}
	print('\n\nEntity Limit Test Count: ' + str(ent_lim))
	print('Sentiment Limit Test Count: ' + str(sent_lim))
	print('Token Limit Test Count: ' + str(tok_lim))

	#generate splits
	fsavef = open('splits', 'w')
	for split_num in range(NUM_SPLITS):
		alldata = range(len(allU))
		random.shuffle(alldata)
		#counters
		sent_has = {i: 0 for i in sent_counts}
		ent_has = {i: 0 for i in ent_counts}
		tok_has = {i: 0 for i in tok_counts}

		print('Split ' + str(split_num))
		nth_split_train = list()
		nth_split_test = list()
		for j in range(len(allU)):
			#counters
			nslots = slots[alldata[j]]
			ent_cur, sent_cur, tok_cur = NLU.getSlotStats([nslots])

			if all([ent_cur[i]+ent_has[i] < ent_lim[i] for i in ent_counts]) and \
			   all([sent_cur[i]+sent_has[i] < sent_lim[i] for i in sent_counts]) and \
			   all([tok_cur[i]+tok_has[i] < tok_lim[i] for i in tok_counts]):
				nth_split_test.append(alldata[j])
				for i in ent_counts:
					ent_has[i] += ent_cur[i]
				for i in sent_counts:
					sent_has[i] += sent_cur[i]
				for i in tok_counts:
					tok_has[i] += tok_cur[i]
			else:
				nth_split_train.append(alldata[j])
		print('Training Instances: ' + str(len(nth_split_train)))
		print('Testing Instances: ' + str(len(nth_split_test)))
		fsavef.write(str(nth_split_train) + ':' + str(nth_split_test) + '\n')
		#print('split(' + str(split_num) + '): ' + str(nth_split_train) + ':' + str(nth_split_test))
	fsavef.flush()
	fsavef.close()

if __name__ == '__main__':
	main()
