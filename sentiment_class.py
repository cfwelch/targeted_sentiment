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
import pickle
#import senti_set
import senti_lexis
import NLU
#import leastSquares
import datetime
import time
#import glove_features

def main():
	#glove_dict = glove_features.makeGlove();
	fo = open("EECS_annotated_samples_anonymized", "r");
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

	cprint("number of utterances: " + str(index));
	cprint("length of lines: " + str(len(sents)));
	cprint("length of targets: " + str(len(targets)));
	cprint("sent 2: " + str(sents[2]));
	cprint("tagset 2: " + str(tagset[2]));

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
	cv.append("692");#bug?
	cprint("vocabulary size: " + str(len(cv)));
	cprint("index of I: " + str(cv.index("i")));
	xtc = [];
	for sent in range(0, len(sents)):
		#print("sentence: " + str(sent));
		#print("s1: " + str(sents[sent]));

		#print(sents[sent] + " - with tagset - " + str(tagset[sent]));
		#dparse = spwrap.parse(sents[sent]);
		#print("DPARSE: " + dparse);

		# add token boundaries to the sentence
		tokenSent = sents[sent];
		for tag in range(0, len(tagset[sent])):
			tokenSent = tokenSent.replace(tagset[sent][tag], " ~~t~~ " + tagset[sent][tag]);
		#print(tokenSent);
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
					#print("parts?: " + str(parts));
					break;
				if part == len(parts) - 1:
					done = True;
		#print("window features: " + str(windowFeatures));

		#print("parts: " + str(parts));
		row = [];
		featureMapG = [[0]*300]*4;
		featureMap = {};
		Nflag = 0;
		for part in range(0, len(parts)):
			#thepart = regex.sub("", parts[part]);
			#thepart = thepart.lower();
			thepart = parts[part].lower();
			theid = cv.index(thepart);
			#print(theid);
			#g_vec = glove_features.getGloveWord(glove_dict, parts[part]);
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
			#for g_vi in range(0, len(g_vec)):
			#	featureMapG[0][g_vi] += g_vec[g_vi];# - mindist/10.0;
			#	featureMapG[1][g_vi] += g_vec[g_vi];# - mindist/10.0;
			#	featureMapG[2][g_vi] += g_vec[g_vi];# - mindist/10.0;
			#	featureMapG[3][g_vi] += g_vec[g_vi];# - mindist/10.0;
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
		# add on the glove features
		for a in range(0, len(featureMapG)):
			temp_vec = [];
			for a_a in range(0, len(featureMapG[a])):
				temp_vec.append(featureMapG[a][a_a]*1.0/len(parts));
			row.extend(temp_vec);
		xtc.append(row);

	#cv = CountVectorizer();
	#xtc = cv.fit_transform(sents);

	#examining data structures here
	#parts = sents[0].split(" ");
	#for part in range(0, len(parts)):
	#	print("PART: " + parts[part]);
	#print("WORD TAKE: " + str(cv.vocabulary_.get(u'i')));
	#print("WORD TAKE: " + str(cv.vocabulary_.get(u'took')));
	#print("WORD DONT: " + str(cv.vocabulary_.get(u'don')));
	#print("WORD DONT: " + str(cv.vocabulary_.get(u't')));
	#print("WORD TAKE: " + str(cv.vocabulary_.get(u'183')));
	#print(str(xtc.shape));
	#print("ROW0");
	#print(xtc[0]);
	#print("ROW1");
	#print(xtc[1]);
	#print("ROW2");
	#print(xtc[2]);
	#print(len(xtc[2]));
	#print(len(xtc));
	#print(type(xtc[0]));
	#print(type(xtc));
	#print(str(len(sents)));
	#endtest

	#xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(xtc, targets, test_size=0.2, random_state=0);

	#use this section of code to do cross validation.
	#shuffle and split into Nfolds parts.
	#testdata = range(0, index);
	#random.shuffle(testdata);
	#folds = list();
	#Nfolds = 10;
	#fsavef = open("folds", "w");
	#for i in range(0, Nfolds):
	#	print("i = " + str(i));
	#	nthfold = testdata[i*index/Nfolds:(i+1)*index/Nfolds];
	#	folds.append(nthfold);
	#	fsavef.write(str(nthfold) + "\n");
	#	print("fold(" + str(i) + "): " + str(nthfold));
	#fsavef.flush();
	#fsavef.close();

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

	#do gridsearch + evaluation
	fscores = open("scores_sentiment", "w");
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
		#likesMatrix, slist = leastSquares.getMatrix(senti_utters);
		# do train-test split
		for j in range(0, len(splits[i][0])):
			#speaker = senti_set.getSpeaker(utterances[splits[i][0][j]][0]);
			#cossim = leastSquares.consineUser(likesMatrix, slist.index(speaker));
			#print("\n" + speaker + ": " + utterances[splits[i][0][j]][0].strip());
			# VECTOR is 38 x 141 -> 264 total
			for LL in range(0, len(sent_to_xtc[splits[i][0][j]])):
				#fvector = likesMatrix[slist.index(speaker)];
				#fvector = fvector.tolist()[0];
				fvector = xtc[sent_to_xtc[splits[i][0][j]][LL]];
				#fvector.append(slist.index(speaker));
				##############################################################
				#entity = tagset[sent_to_xtc[splits[i][0][j]][LL]];
				#entity = tagset2entity(entity);
				#gscore = leastSquares.getGuess(likesMatrix, entity, slist.index(speaker));
				#gscore = leastSquares.getWeightedGuess(cossim, likesMatrix, entity);
				#print("speaker: " + str(speaker) + " - " + str(slist.index(speaker)));
				#fvector.append(gscore);
				########fvector = [gscore];
				##############################################################
				xtrain.append(fvector);
				ytrain.append(targets[sent_to_xtc[splits[i][0][j]][LL]]);
		for j in range(0, len(splits[i][1])):
			#speaker = senti_set.getSpeaker(utterances[splits[i][1][j]][0]);
			#cossim = leastSquares.consineUser(likesMatrix, slist.index(speaker));
			for LL in range(0, len(sent_to_xtc[splits[i][1][j]])):
				#fvector = likesMatrix[slist.index(speaker)];
				#fvector = fvector.tolist()[0];
				fvector = xtc[sent_to_xtc[splits[i][1][j]][LL]];
				#fvector.append(slist.index(speaker));
				##############################################################
				#entity = tagset[sent_to_xtc[splits[i][1][j]][LL]];
				#entity = tagset2entity(entity);
				#gscore = leastSquares.getGuess(likesMatrix, entity, slist.index(speaker));
				#gscore = leastSquares.getWeightedGuess(cossim, likesMatrix, entity);
				#fvector.append(gscore);
				########fvector = [gscore];
				##############################################################
				xtest.append(fvector);
				ytest.append(targets[sent_to_xtc[splits[i][1][j]][LL]]);
		score = 0;


		for gamma in numpy.linspace(0.0001, 0.05, 10):#10steps
			for C in numpy.linspace(0.1, 10, 10):#10steps
				#2 fold
				x1 = xtrain[len(xtrain)/2:];
				x2 = xtrain[:len(xtrain)/2];
				y1 = ytrain[len(ytrain)/2:];
				y2 = ytrain[:len(ytrain)/2];
				x11 = csr_matrix(x1);
				x22 = csr_matrix(x2);
				clf = svm.SVC(gamma=gamma, C=C);
				testout = clf.fit(x1, y1);
				score = clf.score(x2, y2);
				clf = svm.SVC(gamma=gamma, C=C);
				testout = clf.fit(x2, y2);
				score += clf.score(x1, y1);
				score /= 2;
				if score > bestScore:
					bestC = C;
					bestGamma = gamma;
					bestScore = score;
					cprint("Cross Validation Score: " + str(score));
					cprint("Gamma = " + str(gamma) + " and C = " + str(C));

		################ THIS IS FOR CvI EVALUATION ################
		#Ixtest = list();
		#Iytest = list();
		#Cxtest = list();
		#Cytest = list();
		#for j in range(0, len(splits[i][1])):
		#	for LL in range(0, len(sent_to_xtc[splits[i][1][j]])):
		#		fvector = xtc[sent_to_xtc[splits[i][1][j]][LL]];
		#		if coriset[sent_to_xtc[splits[i][1][j]][LL]]:
		#			Cxtest.append(fvector);
		#			Cytest.append(targets[sent_to_xtc[splits[i][1][j]][LL]]);
		#		else:
		#			Ixtest.append(fvector);
		#			Iytest.append(targets[sent_to_xtc[splits[i][1][j]][LL]]);
		#xtrain = csr_matrix(xtrain);
		#Cxtest = csr_matrix(Cxtest);
		#Ixtest = csr_matrix(Ixtest);
		#clf = svm.SVC(gamma=bestGamma, C=bestC);
		#testout = clf.fit(xtrain, ytrain);
		#CBscore = clf.score(Cxtest, Cytest);
		#IBscore = clf.score(Ixtest, Iytest);
		#cprint("Actual Score: " + str(CBscore) + ":" + str(IBscore));
		#fscores.write(str(CBscore) + ":" + str(IBscore) + "\n");
		#fscores.flush();
		###############################################################
		################ THIS IS FOR NORMAL EVALUATION ################
		xtrain = csr_matrix(xtrain);
		xtest = csr_matrix(xtest);
		clf = svm.SVC(gamma=bestGamma, C=bestC);
		testout = clf.fit(xtrain, ytrain);
		bestScore = clf.score(xtest, ytest);
		cprint("Actual Score: " + str(bestScore));
		fscores.write(str(bestScore) + "\n");
		fscores.flush();
		###############################################################
		# save best classifier per fold
		cString = pickle.dumps(clf);
		fsave1 = open("classifiers/sentiment_classifier" + str(i), "w");
		fsave1.write(cString);
		fsave1.flush();
		fsave1.close();
	fscores.close();
	# save feature dictionary
	cvString = pickle.dumps(cv);
	fsave2 = open("sentiment_dictionary", "w");
	fsave2.write(cvString);
	fsave2.flush();
	fsave2.close();

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

def cprint(msg):
	tmsg = msg;
	st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S');
	tmsg = str(st) + ": " + str(tmsg);
	print(tmsg);
	log_file = open("senti_class.log", "a");
	log_file.write(tmsg + "\n");
	log_file.flush();
	log_file.close();

def is_number(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

if __name__ == "__main__":
	main();
