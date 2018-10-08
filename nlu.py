


import pycrfsuite, crf_tagger, entity_distance, dicts, pickle, senti_lexis, sklearn, re

from settings import *
from stanford_corenlp_pywrapper import CoreNLP

def main():
	lines = getALines()
	# format folds inputs
	fsplits = open("splits")
	slines = fsplits.readlines()
	splits = list()
	for i in range(0, len(slines)):
		parts = slines[i].strip().split(":")
		train = list()
		test = list()
		for s in parts[0][1:-1].split(", "):
			train.append(int(s))
		for s in parts[1][1:-1].split(", "):
			test.append(int(s))
		splits.append((train, test))
	fsplits.close()

	print("Number of folds: " + str(NUM_SPLITS))
	fdict = open("sentiment_dictionary", "r")
	cv = pickle.loads(fdict.read())
	fdict.close()
	foutput = open("nlu_scores", "w")

	for fold in range(0, NUM_SPLITS):
		#for evaluation
		scores = {j: {i: 0 for i in ['correct', 'guessed', 'actual']} for j in ENT_TYPES}
		#get utterances
		in_utter = getUtterances(lines)
		take_utter = list()
		for i in range(0, len(in_utter)):
			if i in splits[fold][1]:
				take_utter.append(in_utter[i])
		in_utter = take_utter

		fclf = open("classifiers/sentiment_classifier" + str(fold), "r")
		clf = pickle.loads(fclf.read())
		fclf.close()

		proc = CoreNLP("pos", corenlp_jars=[PATH_TO_STANFORD_CORENLP])
		tagger = pycrfsuite.Tagger()
		tagger.open("taggers/advising_crf_tagger" + str(fold))

		#classify utterances
		for k in range(0, len(in_utter)):
			print("Current Utterance: " + in_utter[k][0])
			#get slots from utterance
			slots = getSlots(in_utter[k])
			print("Slots: " + str(slots))
			#constituency parse
			parsed = proc.parse_doc(in_utter[k][0])
			#print(parsed)
			#print(str(list(parsed['sentences'][0]['tokens'])))
			print("\n\n\n")
			print("Number of parsed sentences: " + str(len(parsed['sentences'])))
			spos_tlist = list()
			for i in range(0, len(parsed['sentences'])):
				spos_tuples = zip(parsed['sentences'][i]['tokens'], parsed['sentences'][i]['pos'])
				spos_tlist.append(spos_tuples)
			X_test = [crf_tagger.sent2featuresWithSent(s, in_utter[k][0]) for s in spos_tlist]
			y_pred = [tagger.tag(xseq) for xseq in X_test]

			print(parsed['sentences'][0]['tokens'])
			print(y_pred[0])

			ent_list = {i: [] for i in ENT_TYPES}
			for i in range(0, len(parsed['sentences'])):
				etemp = getEntities(parsed['sentences'][i]['tokens'], y_pred[i])
				for etype in ENT_TYPES:
					ent_list[etype].extend(etemp[etype])

			for i in ENT_TYPES:
				print(i + ': ' + str(ent_list[i]))

			ent_outs = {i: [] for i in ENT_TYPES}
			for etype in ENT_TYPES:
				for i in range(len(ent_list[etype])):
					ent_outs[etype].append(getClassLabel(in_utter[k][0], ent_list[etype][i], y_pred[0], parsed['sentences'][0]['tokens'], cv, clf))

			for etype in ENT_TYPES:
				#generate tuples for comparison of classes
				tlist = list()
				seval = list()
				for q in range(len(slots[etype])):
					scores[etype]['actual'] += 1
					ent_t = {k: v for k,v in slots[etype][q].items() if k in ENT_TYPES[etype]}
					tlist.append(ent_t)
					seval.append(slots[etype][q]['sentiment'])
				for i in range(len(ent_outs[etype])):
					scores[etype]['guessed'] += 1
					if ent_list[etype][i] in tlist:
						if seval[tlist.index(ent_list[etype][i])] == ent_outs[etype][i]:
							scores[etype]['correct'] += 1

			print('current scores: ' + str(scores))

			#print output
			print("\n\nInput: " + in_utter[k][0])
			print("Output: ")
			for etype in ENT_TYPES:
				for i in range(len(ent_list[etype])):
					print(etype + ': ' + str(ent_list[etype][i]) + " - " + str(ent_outs[etype][i]))

		precision = sum([scores[i]['correct'] for i in ENT_TYPES]) * 1.0 / sum([scores[i]['guessed'] for i in ENT_TYPES])
		recall = sum([scores[i]['correct'] for i in ENT_TYPES]) * 1.0 / sum([scores[i]['actual'] for i in ENT_TYPES])
		ith_row = [precision, recall]

		for i in ENT_TYPES:
			tprecision = scores[i]['correct'] * 1.0 / scores[i]['guessed']
			trecall = scores[i]['correct'] * 1.0 / scores[i]['actual']
			ith_row.append(tprecision)
			ith_row.append(trecall)

		foutput.write(str(ith_row) + '\n')
	foutput.close()

def getUtterances(lines):
	full = list()
	temp = list()
	for i in range(0, len(lines)):
		if "\n" == lines[i]:
			full.append(temp)
			temp = list()
		else:
			temp.append(lines[i])
	print("Utterances Parsed: " + str(len(full)))
	return full

def getALines():
	fo = open("EECS_annotated_samples_anonymized", "r")
	lines = fo.readlines()
	fo.close()
	return lines

# takes input from getUtterances
def getTextLines(utterances):
	tlines = []
	for line in utterances:
		tline = line[0].strip()
		tline = re.sub(r'([eE][eE][cC][sS])(\d+)', '\\1 \\2', tline)
		tlines.append(tline)
	return tlines

# uselink is False for NLU testing, True for sentiment testing
def getSlots(utterance, uselink=False):
	ents = {i: [] for i in ENT_TYPES}
	mode = ''

	cur_ent = {}
	for i in range(1, len(utterance)):
		if utterance[i].endswith('\n'):
			utterance[i] = utterance[i][:-1]
		#print("U:" + str(utterance[i]))

		for ent_type in ENT_TYPES:
			if utterance[i].startswith('<' + ent_type):
				mode = ent_type

		if len(utterance[i].split('=')) == 2:
			attparts = utterance[i].strip().split('=')
			if attparts[1].endswith('>'):
				attparts[1] = attparts[1][:-1]
			cur_ent[attparts[0]] = attparts[1]

		if utterance[i].endswith('>'):
			# print("Link: " + str(cur_ent['link']))
			if 'sentiment' not in cur_ent:
				cur_ent['sentiment'] = 'neutral'

			ents[mode].append(cur_ent)
			cur_ent = {}
	return ents

def getSlotStats(slots):
	ent_counts = {i: 0 for i in ENT_TYPES}
	sent_counts = {i: 0 for i in ['positive', 'negative', 'neutral']}
	tok_counts = {i+'_'+j+'_'+k: 0 for i in ENT_TYPES for j in ENT_TYPES[i] for k in ['B', 'I']}
	for slot in slots:
		for i in ENT_TYPES:
			ent_counts[i] += len(slot[i])
			for tent in slot[i]:
				sent_counts[tent['sentiment']] += 1
				for j in ENT_TYPES[i]:
					if j not in tent:
						continue
					attribute = tent[j].split()
					for k in range(len(attribute)):
						tok_counts[i+'_'+j+'_'+('B' if k == 0 else 'I')] += 1

	# trim zero tokens if you are getting the whole list
	if len(slots) > 1:
		tok_counts = {k: v for k,v in tok_counts.items() if v > 0}

	return ent_counts, sent_counts, tok_counts

def getClassLabel(utterance, entity, BIO, tokens, cv, clf):
	regex = re.compile(r"[^a-zA-Z0-9_\~ ]+")

	# add token boundaries to the sentence
	tokenSent = utterance
	# use the ID, not the department...
	for tag in entity:
		if "" != entity[tag]:
			print("eT:" + str(entity[tag]))
			tokenSent = tokenSent.replace(entity[tag], " ~~t~~ " + entity[tag])
	#print(tokenSent)
	parts = regex.sub("", tokenSent).split(" ")

	# remove empty parts from the sentence
	while "" in parts:
		parts.remove("")

	# locate window feature indicies
	windowFeatures = []
	done = False
	while not done:
		for part in range(0, len(parts)):
			if "~~t~~" == parts[part]:
				windowFeatures += [part]
				parts.remove(parts[part])
				#print("parts?: " + str(parts))
				break
			if part == len(parts) - 1:
				done = True
	print("window features: " + str(windowFeatures))

	for i in range(0, len(tokens)):
		tokens[i] = regex.sub("", tokens[i])
	row = []
	featureMapG = [[0]*300]*4
	featureMap = {}
	Nflag = 0
	for part in range(0, len(tokens)):
		thepart = tokens[part].lower()
		if thepart in cv:
			theid = cv.index(thepart)
			mindist = 999
			for wf in range(0, len(windowFeatures)):
				distance = abs(windowFeatures[wf] - part)
				if distance < mindist:
					mindist = distance
			mindist += 1
			sentiz = senti_lexis.lexCounts(thepart)
			#for g_vi in range(0, len(g_vec)):
			#	featureMapG[0][g_vi] += g_vec[g_vi]# - mindist/10.0
			#	featureMapG[1][g_vi] += g_vec[g_vi]# - mindist/10.0
			#	featureMapG[2][g_vi] += g_vec[g_vi]# - mindist/10.0
			#	featureMapG[3][g_vi] += g_vec[g_vi]# - mindist/10.0
			if theid in featureMap:
				# 1.0 - mindist / 10.0 worked well for the first distance measure...
				# featureMap[theid] += 1.0 / mindist
				featureMap[theid][0] += 1.0 - mindist / 10.0
				featureMap[theid][1] += (1.0 - mindist / 10.0) * sentiz[0]
				featureMap[theid][2] += (1.0 - mindist / 10.0) * sentiz[1]
				featureMap[theid][3] += (1.0 - mindist / 10.0) * sentiz[2]
				if Nflag > 0:
					featureMap[theid][4] = 1.0
			else:
				# featureMap[theid] = 1.0 / mindist
				# count, positive, negative, neutral, negate
				featureMap[theid] = [0, 0, 0, 0, 0]
				featureMap[theid][0] = 1.0 - mindist / 10.0
				featureMap[theid][1] = (1.0 - mindist / 10.0) * sentiz[0]
				featureMap[theid][2] = (1.0 - mindist / 10.0) * sentiz[1]
				featureMap[theid][3] = (1.0 - mindist / 10.0) * sentiz[2]
				if Nflag > 0:
					featureMap[theid][4] = 1.0
			if Nflag > 0:
				Nflag -= 1
			if senti_lexis.lexNegate(thepart):
				Nflag = 2
	for i in range(0, len(cv)):
		if i in featureMap:
			row.extend(featureMap[i])
		else:
			row.extend([0, 0, 0, 0, 0])
	# make prediction
	return clf.predict([row])

def getEntities(tokens, BIO):
	ent_list = {i: [] for i in ENT_TYPES}
	cur_ent = None
	cur_type = None

	tcount = 0
	for label in BIO:
		lparts = label.split('_')
		etype = lparts[0] if len(lparts) == 3 else None
		attribute = lparts[1] if len(lparts) == 3 else None
		bio_part = lparts[2] if len(lparts) == 3 else lparts[0]

		if bio_part != 'O':
			if (cur_ent != None and (etype != cur_type or (attribute in cur_ent and bio_part == 'B'))) or cur_ent == None:
				if cur_ent != None:
					ent_list[cur_type].append(cur_ent)
				cur_ent = {}
				cur_type = etype

			if bio_part == 'B':
				cur_ent[attribute] = tokens[tcount]
			elif bio_part == 'I':
				cur_ent[attribute] += ' ' + tokens[tcount]

		tcount += 1

	if cur_ent != None:
		ent_list[cur_type].append(cur_ent)
	
	for i in ENT_TYPES:
		tlist = []
		for j in ent_list[i]:
			if j not in tlist:
				tlist.append(j)
		ent_list[i] = tlist

	return ent_list

def is_number(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

if __name__ == "__main__":
	main()
