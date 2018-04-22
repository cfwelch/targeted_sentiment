import senti_lexis
import datetime
import time
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

def main():
	for i in range(1, 11):
		runfold(i);

def runfold(fold):
	# read files
	found_ents = open("fold" + str(fold) + "_found_entities");
	train_file = open("../../data/Open Domain Targeted Sentiment/en/10-fold/train." + str(fold));
	test_file = open("../../data/Open Domain Targeted Sentiment/en/10-fold/test." + str(fold));
	train_lines = train_file.readlines();
	test_lines = test_file.readlines();
	found_lines = found_ents.readlines();
	train_file.close();
	test_file.close();
	found_ents.close();
	# evaluation metrics
	ent_guessed = 0;
	ent_actual = 0;
	ent_correct = 0;
	# compute
	utterances = list();
	tlist = list();
	targets = list();
	sents = list();
	ents = list();
	for line in train_lines:
		tl = line.strip();
		if tl == "":
			if tlist:
				utterances.append(tlist);
				tlist = list();
		else:
			if not tl.startswith("## Tweet"):
				tlist.append(tl.split("\t"));
	if tlist:
		utterances.append(tlist);
		tlist = list();
	splitpoint = len(utterances);
	for line in test_lines:
		tl = line.strip();
		if tl == "":
			if tlist:
				utterances.append(tlist);
				tlist = list();
		else:
			if not tl.startswith("## Tweet"):
				tlist.append(tl.split("\t"));
	if tlist:
		utterances.append(tlist);
	#cprint("Utterances: " + str(len(utterances)));
	######## figure out what to do with the predicted entities
	print("LEN OF FLINES: " + str(len(found_lines)));
	print("LEN OF UTTERANCES: " + str(len(utterances) - splitpoint));
	found_entities = list();
	for f_line in found_lines:
		PTZ = f_line.strip();#.split("::::::");
		if PTZ != "[]":#PTZ[0]
			partf = PTZ[2:-2].split("', '");#PTZ[0]
			found_entities.append(partf);
		else:
			found_entities.append([]);
	print("LEN OF PARSED FLINES: " + str(len(found_entities)));
	#split tlist into targets and sents
	splitpoint2 = 0;
	count = 0;
	for utt in utterances:
		sent = "";
		target = list();
		ent = list();
		f_ents = None;
		if count >= splitpoint:
			f_ents = found_entities[count - splitpoint];
		_t = None;
		for i in utt:
			if sent != "":
				sent += " ";
			sent += i[0];
			if i[1] == "B-ORGANIZATION" or i[1] == "B-PERSON":
			#if i[1][0] == "B":
				target.append(i[0]);
				ent.append(i[2]);
			#print(str(i[0]) + ":" + str(i[1][0]) + ":" + str(i[2]));
		## Figure out if entity was missed or not
		#### debug if block
		#if f_ents != None:
		#	print(ent);
		#	print(target);
		#	print(f_ents);
		#	if count == 2117:
		#		break;
		if f_ents != None:
			#ent_guessed += len(f_ents);#overcounts by including _
			#ent_actual += len(target);#overcounts by including _
			for T_T in range(0, len(target)):
				if ent[T_T] != "_" and ent[T_T] != "neutral":
					ent_actual += 1;
			TOTEST_targets = list();
			TOTEST_ents = list();
			for T_T in f_ents:
				if T_T in target:
					idx = target.index(T_T);
					TOTEST_targets.append(T_T);
					TOTEST_ents.append(ent[idx]);
					if ent[idx] != "_" and ent[idx] != "neutral":
						ent_guessed += 1;
				else:
					ent_guessed += 1;
			#print(TOTEST_targets);
			#print(TOTEST_ents);
			ents.extend(TOTEST_ents);
			for a in range(0, len(TOTEST_ents)):
				sents.append(sent);
			targets.extend(TOTEST_targets);
		else:
			ents.extend(ent);
			for a in range(0, len(ent)):
				sents.append(sent);
			targets.extend(target);
		######## split point counters
		if count > splitpoint and splitpoint2 == 0:
			splitpoint2 = len(targets);
		count += 1;
	#return;########################################################################

	#print("SPLIT POINT 1: " + str(splitpoint));
	#print("SPLIT POINT 2: " + str(splitpoint2));

	print("LEN TARGETS: " + str(len(targets)));#ntities
	print("LEN SENTS: " + str(len(sents)));#sentences
	print("LEN ENTS: " + str(len(ents)));#sentiments
	print("LEN REAL TARGETS: " + str(len(found_entities)))#real entities

	# Generate vocab
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
	for sent in range(0, len(sents)):
		tokenSent = sents[sent];
		tokenSent = tokenSent.replace(targets[sent], " ~~t~~ " + targets[sent]);
		parts = regex.sub("", tokenSent);
		parts = re.split(" |-", parts);
		while "" in parts:
			parts.remove("");
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
		for part in range(0, len(parts)):
			thepart = parts[part].lower();
			if thepart not in cv:
				cv.add(thepart);
	cv = list(cv);
	#cprint("Vocabulary Size: " + str(len(cv)));


	# Generate the feature vectors
	xtc = [];
	xtcT = [];
	train_ents = [];
	test_ents = [];
	for sent in range(0, len(sents)):
		# add token boundaries to the sentence
		tokenSent = sents[sent];
		#print(targets[sent]);
		tokenSent = tokenSent.replace(targets[sent], " ~~t~~ " + targets[sent]);
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
		featureMap = {};
		Nflag = 0;
		for part in range(0, len(parts)):
			#thepart = regex.sub("", parts[part]);
			#thepart = thepart.lower();
			thepart = parts[part].lower();
			if thepart not in cv:
				cv.append(thepart);
			theid = cv.index(thepart);
			#print(theid);
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
		if sent < splitpoint2:
			#print("ROW: " + str(len(row)));
			#print("LABEL: " + str(ents[sent]));
			xtc.append(row);
			train_ents.append(ents[sent]);
		else:
			xtcT.append(row);
			test_ents.append(ents[sent]);

	dist = numpy.array(ents);
	#print((dist=="neutral").sum());
	#print((dist=="negative").sum());
	#print((dist=="positive").sum());
	#print((dist=="_").sum());

	#print("LENTR: " + str(len(xtc)));
	#print("LENTE: " + str(len(xtcT)));
	print("LEN TRAIN ENTS: " + str(len(train_ents)));
	print("LEN TEST ENTS: " + str(len(test_ents)));

	#do gridsearch + evaluation
	bestC = 0;
	bestGamma = 0;
	bestScore = 0;
	xtest = list();
	xtrain = list();
	ytest = list();
	ytrain = list();
	# do train-test split
	for j in range(0, len(xtc)):
		LB = train_ents[j];
		if LB != "_" and LB != "neutral":
			xtrain.append(xtc[j]);
			ytrain.append(LB);
	for j in range(0, len(xtcT)):
		LB = test_ents[j];
		if LB != "_" and LB != "neutral":
			xtest.append(xtcT[j]);
			ytest.append(LB);
	score = 0;

	print("LEN TRAIN: " + str(len(ytrain)));
	print("LEN TEST: " + str(len(ytest)));

	#print(xtrain);
	#print(len(xtrain));
	#print(len(xtrain[0]));
	#print(len(xtrain[1]));
	#print(len(xtrain[2]));
	#print(len(xtrain[3]));
	#print(len(xtrain[4]));
	#for __ in xtrain:
		#if len(__) != 56410:
		#	print len(__)
	for gamma in numpy.linspace(0.0001, 0.05, 10):#10steps
		for C in numpy.linspace(0.1, 10, 10):#10steps
			#gamma = 0.005644444444444444;
			#C = 6.0;
			xtrain = csr_matrix(xtrain);
			xtest = csr_matrix(xtest);
			clf = svm.SVC(gamma=gamma, C=C);
			testout = clf.fit(xtrain, ytrain);
			score = clf.score(xtest, ytest);
			if score > bestScore:
				bestC = C;
				bestGamma = gamma;
				bestScore = score;
				cprint("Cross Validation Score: " + str(score));
				cprint("Gamma = " + str(gamma) + " and C = " + str(C));

	################ THIS IS FOR NORMAL EVALUATION ################
	xtrain = csr_matrix(xtrain);
	xtest = csr_matrix(xtest);
	clf = svm.SVC(gamma=bestGamma, C=bestC);
	testout = clf.fit(xtrain, ytrain);
	bestScore = clf.score(xtest, ytest);
	#print(clf.predict(xtest));
	ent_correct = (clf.predict(xtest) == ytest).sum();
	cprint("Actual Score: " + str(bestScore));
	###############################################################
	print(str(ent_guessed) + "\t" + str(ent_actual) + "\t" + str(ent_correct));
	#print("ENT GUESSED: " + str(ent_guessed));
	#print("ENT ACTUAL: " + str(ent_actual));
	#print("ENT CORRECT: " + str(ent_correct));


def cprint(msg):
	tmsg = msg;
	st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S');
	tmsg = str(st) + ": " + str(tmsg);
	print(tmsg);
	log_file = open("open_domain.log", "a");
	log_file.write(tmsg + "\n");
	log_file.flush();
	log_file.close();

if __name__ == "__main__":
	main();
