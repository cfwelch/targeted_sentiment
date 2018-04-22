import sys
import numpy
import re
import string
import spwrap
import random
from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from scipy.sparse import csr_matrix
import numpy as np
import pickle
import senti_set
import senti_lexis
import NLU
import leastSquares

def main():
	fo = open("../data/extract_samples/EECS_annotated_samples", "r");
	lines = fo.readlines();
	utterances = NLU.getUtterances(lines);
	mode = False;
	sents = list();
	targets = list();
	lastTaken = "";
	lastSent = "";
	isclass = False;
	tagset = list();
	coriset = list();
	lastTagset = list();
	index = 0;
	# to make cross validation work after sentences are duplicated for entities
	sent_to_xtc = dict();
	sent_to_xtc[0] = list();
	for i in range(len(lines)):
		data = lines[i].strip();
		if "" == data:
			index += 1;
			sent_to_xtc[index] = list();
		if data.startswith("<class") or data.startswith("<instructor"):
			mode = True;
			lastTaken = "";
			lastTagset = list();
		if data.startswith("<class"):
			isclass = True;
		if mode and data.startswith("sentiment="):
			lastTaken = data[10:];
			if lastTaken.endswith(">"):
				lastTaken = lastTaken[:-1];
		if mode and data.startswith("name="):
			temp = data[5:];
			if temp.endswith(">"):
				temp = temp[:-1];
			lastTagset.append(temp);
		if mode and data.startswith("id="):
			temp = data[3:];
			if temp.endswith(">"):
				temp = temp[:-1];
			lastTagset.append(temp);
		if mode and data.startswith("department="):
			temp = data[11:];
			if temp.endswith(">"):
				temp = temp[:-1];
			lastTagset.append(temp);
		if not mode and "" != data:
			lastSent = data;
		if data.endswith(">"):
			mode = False;
			coriset.append(isclass);
			isclass = False;
			sents.append(lastSent);
			tagset.append(lastTagset);
			sent_to_xtc[index].append(len(sents)-1);
			if lastTaken == "":
				targets.append("neutral");
			else:
				targets.append(lastTaken);

	# This will print out mapping from sentences to entity vectors (XTC)
	#foutest = open("outtestJ", "w");
	#for key in sent_to_xtc:
	#	foutest.write(str(key) + " : " + str(sent_to_xtc[key]) + "\n");
	#foutest.flush();
	#foutest.close();

	#randomly sample utterances
	#testdata = random.sample(range(0, index), index/5);

	print("number of utterances: " + str(index));
	print("length of lines: " + str(len(sents)));
	print("length of targets: " + str(len(targets)));
	print("sent 2: " + str(sents[2]));
	print("tagset 2: " + str(tagset[2]));

	cv = set();
	regex = re.compile(r"[^a-zA-Z0-9_\~\- ]+");
	for sent in range(0, len(sents)):
		parts = sents[sent].split(" ");
		for part in range(0, len(parts)):
			thepart = regex.sub("", parts[part]);
			# corner case for hyphens
			hps = thepart.split("-");
			if len(hps) > 1:
				for hi in range(0, len(hps)):
					cv.add(hps[hi].lower());
			# end corner case for hyphens
			thepart = thepart.lower();
			cv.add(thepart);
	cv = list(cv);
	cv.append("452");#bug?
	print("vocabulary size: " + str(len(cv)));
	print("index of I: " + str(cv.index("i")));
	xtc = [];
	for sent in range(0, len(sents)):
		print("sentence: " + str(sent));
		print("s1: " + str(sents[sent]));

		#print(sents[sent] + " - with tagset - " + str(tagset[sent]));
		#dparse = spwrap.parse(sents[sent]);
		#print("DPARSE: " + dparse);

		# add token boundaries to the sentence
		tokenSent = sents[sent];
		for tag in range(0, len(tagset[sent])):
			tokenSent = tokenSent.replace(tagset[sent][tag], " ~~t~~ " + tagset[sent][tag]);
		print(tokenSent);
		parts = regex.sub("", tokenSent);
		# this handles split and hyphen corner case
		parts = re.split(" |-", parts);

		# remove empty parts from the sentence
		while "" in parts:
			parts.remove("");

		# locate window feature indicies
		windowFeatures = [];
		done = False;
		while not done:
			for part in range(0, len(parts)):
				if "~~t~~" == parts[part]:
					windowFeatures += [part];
					parts.remove(parts[part]);
					print("parts?: " + str(parts));
					break;
				if part == len(parts) - 1:
					done = True;
		print("window features: " + str(windowFeatures));

		print("parts: " + str(parts));
		row = [];
		featureMap = {};
		Nflag = 0;
		for part in range(0, len(parts)):
			#thepart = regex.sub("", parts[part]);
			#thepart = thepart.lower();
			thepart = parts[part].lower();
			theid = cv.index(thepart);
			print(theid);
			mindist = 999;
			for wf in range(0, len(windowFeatures)):
				##############################################################
				## This is the distance measure for window linear distance!
				distance = abs(windowFeatures[wf] - part);
				##############################################################
				## This is the distance measure for dependency tree distnace!
				## distance = spwrap.treeDistance(parts[windowFeatures[wf]], parts[part], dparse);
				##############################################################
				if distance < mindist:
					mindist = distance;
			mindist += 1;
			sentiz = senti_lexis.lexCounts(thepart);
			if theid in featureMap:
				# 2.0 - mindist / 7.0 worked well for the first distance measure...
				# featureMap[theid] += 1.0 / mindist;
				featureMap[theid][0] += 2.0 - mindist / 7.0;
				featureMap[theid][1] += (2.0 - mindist / 7.0) * sentiz[0];
				featureMap[theid][2] += (2.0 - mindist / 7.0) * sentiz[1];
				featureMap[theid][3] += (2.0 - mindist / 7.0) * sentiz[2];
				if Nflag > 0:
					featureMap[theid][4] = 1.0;
			else:
				# featureMap[theid] = 1.0 / mindist;
				# count, positive, negative, neutral, negate
				featureMap[theid] = [0, 0, 0, 0, 0];
				featureMap[theid][0] = 2.0 - mindist / 7.0;
				featureMap[theid][1] = (2.0 - mindist / 7.0) * sentiz[0];
				featureMap[theid][2] = (2.0 - mindist / 7.0) * sentiz[1];
				featureMap[theid][3] = (2.0 - mindist / 7.0) * sentiz[2];
				if Nflag > 0:
					featureMap[theid][4] = 1.0;
			if Nflag > 0:
				Nflag -= 1;
			if senti_lexis.lexNegate(thepart):
				Nflag = 2;
		for i in range(0, len(cv)):
			if i in featureMap:
				row.extend(featureMap[i]);
			else:
				row.extend([0, 0, 0, 0, 0]);
		xtc.append(row);

	#instead read the data from splits file
	fsplits = open("splits");
	lines = fsplits.readlines();
	splits = list();
	for i in range(0, len(lines)):
		parts = lines[i].strip().split(":");
		train = list();
		test = list();
		for s in parts[0][1:-1].split(", "):
			train.append(int(s));
		for s in parts[1][1:-1].split(", "):
			test.append(int(s));
		splits.append((train, test));
	fsplits.close();
	#test print the first split
	#print(splits[0][0]);
	#print(splits[0][1]);

	bestsplit = -1;
	BSscore = 0;
	for i in range(0, len(splits)):
		bestC = 0;
		bestGamma = 0;
		bestScore = 0;
		xtest = list();
		xtrain = list();
		ytest = list();
		ytrain = list();
		# add the utterance set generation here for senti_set
		senti_utters = list();
		for j in range(0, len(splits[i][0])):
			senti_utters.append(utterances[splits[i][0][j]]);
		likesMatrix, slist = leastSquares.getMatrix(senti_utters);
		# do train-test split
		csims = np.array([0.0]*38);
		totz = 0;
		#for j in range(0, len(splits[i][0])):
		#	speaker = senti_set.getSpeaker(utterances[splits[i][0][j]][0]);
		#	cossim = leastSquares.cosineUserWE(likesMatrix, slist.index(speaker));
		#	np.add(csims, cossim);
		#	totz += 1;
		for j in range(0, len(splits[i][1])):
			speaker = senti_set.getSpeaker(utterances[splits[i][1][j]][0]);
			cossim = leastSquares.cosineUserWE(likesMatrix, slist.index(speaker));
			cossim = np.array(cossim);
			csims = np.add(csims, cossim);
			totz += 1;
		for j in range(0, len(csims)):
			csims[j] /= totz;
		print(csims.tolist());

def tagset2entity(tagset):
	entity = "";
	for i in range(0, len(tagset)):
		if tagset[i].lower() == "eecs":
			pass;
		elif is_number(tagset[i]):
			entity = int(tagset[i]);
		else:
			if not is_number(entity):
				entity = tagset[i];
	return entity;

def is_number(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

if __name__ == "__main__":
	main();
