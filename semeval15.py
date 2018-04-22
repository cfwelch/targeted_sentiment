import numpy
import re
import string
import random
from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from scipy.sparse import csr_matrix
import senti_lexis

def main():
	trainExamples = loadExamples("../data/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml");
	testExamples = loadExamples("../data/ABSA15_Restaurants_Test.xml");
	print("Train Examples: " + str(len(trainExamples)));
	print("Test Examples: " + str(len(testExamples)));
	# generate vocabulary
	cv = set();
	fullset = list();
	fullset.extend(trainExamples);
	fullset.extend(testExamples);
	regex = re.compile(r"[^a-zA-Z0-9_\~\- ]+");
	for sent in range(0, len(fullset)):
		parts = fullset[sent][0].split(" ");
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

	# generate aspect set
	aset = set();
	for sent in range(0, len(fullset)):
		aset.add(fullset[sent][4]);
	alist = list();
	alist.extend(aset);
	print("Aspect Set: " + str(alist));

	# generate feature vectors
	xtrain = featureVector(cv, trainExamples);
	xtest = featureVector(cv, testExamples);

	bestC = 0;
	bestGamma = 0;
	bestScore = 0;
	ytest = list();
	ytrain = list();
	# do train-test split for Y
	for i in range(0, len(trainExamples)):
		ytrain.append(trainExamples[i][2]);
	for i in range(0, len(testExamples)):
		ytest.append(testExamples[i][2]);

	for gamma in numpy.linspace(0.0001, 0.05, 20):#10steps
		for C in numpy.linspace(0.1, 10, 20):#10steps
			#K fold
			K = 3;
			folds = list();
			score = 0;
			for i in range(0, K):
				x1 = list();
				x2 = list();
				y1 = list();
				y2 = list();
				chunk = len(xtrain)/K;
				for j in range(0, K):
					if i == j:
						x2.extend(xtrain[i*chunk:(i+1)*chunk]);
						y2.extend(ytrain[i*chunk:(i+1)*chunk]);
					else:
						x1.extend(xtrain[i*chunk:(i+1)*chunk]);
						y1.extend(ytrain[i*chunk:(i+1)*chunk]);
				x11 = csr_matrix(x1);
				x22 = csr_matrix(x2);
				clf = svm.SVC(gamma=gamma, C=C);
				testout = clf.fit(x1, y1);
				score += clf.score(x2, y2);
			score /= K;
			if score > bestScore:
				bestC = C;
				bestGamma = gamma;
				bestScore = score;
				print("Cross Validation Score: " + str(score));
				print("Gamma = " + str(gamma) + " and C = " + str(C));
	
	xtrain = csr_matrix(xtrain);
	xtest = csr_matrix(xtest);
	clf = svm.SVC(gamma=bestGamma, C=bestC);
	testout = clf.fit(xtrain, ytrain);
	bestScore = clf.score(xtest, ytest);
	print("Actual Score: " + str(bestScore));

def featureVector(cv, examples):
	regex = re.compile(r"[^a-zA-Z0-9_\~\- ]+");
	xtc = [];
	for sent in range(0, len(examples)):
		print("sentence: " + str(sent));
		print("s1: " + str(examples[sent][0]));

		# add token boundaries to the sentence
		tokenSent = examples[sent][0];
		parts = regex.sub("", tokenSent);
		# this handles split and hyphen corner case
		parts = re.split(" |-", parts);
		# remove empty parts from the sentence
		while "" in parts:
			parts.remove("");
		print(parts);

		# locate window feature indicies
		windowFeatures = [];
		if examples[sent][3] != "NULL":
			inToken = True;
			tokenNumber = 0;
			for i in range(0, int(examples[sent][1])):
				if examples[sent][0][i] == " " or examples[sent][0][i] == "-":
					if inToken:
						tokenNumber += 1;
					inToken = False;
				else:
					inToken = True;
			print(str(tokenNumber) + ":" + parts[tokenNumber]);
			windowFeatures = [tokenNumber];

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
				if distance < mindist:
					mindist = distance;
			mindist += 1;
			if len(windowFeatures) == 0:
				mindist = 7;
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
	return xtc;

def loadExamples(fpath):
	restF = open(fpath);
	lines = restF.readlines();
	restF.close();
	lastSent = "";
	lastOpinion = "";
	examples = list();
	for i in lines:
		sent = i.strip();
		if sent.startswith("<text>"):
			lastSent = sent[6:-7];
		elif sent.startswith("<Opinion "):
			polarity = sent[sent.index("polarity")+10:];
			polarity = polarity[:polarity.index("\"")];
			fromInd = sent[sent.index("from=")+6:];
			fromInd = fromInd[:fromInd.index("\"")];
			target = sent[sent.index("target=")+8:];
			target = target[:target.index("\"")];
			category = sent[sent.index("category=")+10:];
			category = category[:category.index("\"")];
			examples.append([lastSent, fromInd, polarity, target, category]);
	return examples;

if __name__ == "__main__":
	main();
