from stanford_corenlp_pywrapper import CoreNLP
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
import sklearn
import dicts
import numpy
import entity_distance
import pycrfsuite
import re
import acronyms
import time
import datetime
import LDAlda

def main():
	for i in range(1, 11):
		runfold(i);

def runfold(fold):
	bioset = ['B-URL', 'I-DATE', 'B-EMAIL', 'B-MONEY', 'B-PLACE', 'I-PLACE', 'I-ORGANIZATION', 'I-TIME', 'I-URL', 'I-PERSON', 'B-TIME', 'I-PERCENT', 'B-PERCENT', 'B-PERSON', 'O', 'I-MONEY', 'B-DATE', 'B-TELEPHONE', 'B-ORGANIZATION', 'I-EMAIL', 'I-TELEPHONE'];
	#bioset = ['B-URL', 'I-DATE', 'B-EMAIL', 'B-MONEY', 'B-PLACE', 'I-PLACE', 'I-ORGANIZATION', 'I-TIME', 'I-URL', 'I-PERSON', 'B-TIME', 'I-PERCENT', 'B-PERCENT', 'B-PERSON', 'O', 'I-MONEY', 'B-DATE', 'B-TELEPHONE', 'B-ORGANIZATION', 'I-EMAIL', 'I-TELEPHONE'];
	train_file = open("../../data/Open Domain Targeted Sentiment/en/10-fold/train." + str(fold));
	test_file = open("../../data/Open Domain Targeted Sentiment/en/10-fold/test." + str(fold));
	train_lines = train_file.readlines();
	test_lines = test_file.readlines();
	train_file.close();
	test_file.close();
	train_utterances = list();
	test_utterances = list();
	tlist = list();
	train_sents = list();
	test_sents = list();
	labe_set = set();
	for line in train_lines:
		tl = line.strip();
		if tl == "":
			if tlist:
				train_utterances.append(tlist);
				tlist = list();
		else:
			if not tl.startswith("## Tweet"):
				tl_L = tl.split("\t")[:3];
				labe_set.add(tl_L[1]);
				tlist.append(tl_L);
	if tlist:
		train_utterances.append(tlist);
		tlist = list();
	for line in test_lines:
		tl = line.strip();
		if tl == "":
			if tlist:
				test_utterances.append(tlist);
				tlist = list();
		else:
			if not tl.startswith("## Tweet"):
				tl_L = tl.split("\t")[:3];
				labe_set.add(tl_L[1]);
				tlist.append(tl_L);
	if tlist:
		test_utterances.append(tlist);
	print(labe_set);
	cprint("Train Utterances: " + str(len(train_utterances)));
	cprint("Test Utterances: " + str(len(test_utterances)));
	#split tlist into sents
	count = 0;
	for utt in train_utterances:
		sent = "";
		for i in utt:
			if sent != "":
				sent += " ";
			sent += i[0];
		train_sents.append(sent);
	for utt in test_utterances:
		sent = "";
		for i in utt:
			if sent != "":
				sent += " ";
			sent += i[0];
		test_sents.append(sent);
	cprint("Train Sents: " + str(len(train_sents)));
	cprint("Test Sents: " + str(len(test_sents)));

	################# GET POS TAGS
	proc = CoreNLP("pos", corenlp_jars=["/home/cfwelch/workspace/stanford-corenlp-full-2015-12-09/*"]);
	for j in range(0, len(train_utterances)):
		parsed = proc.parse_doc(train_sents[j]);
		#print(train_utterances[j]);
		d_a = parsed["sentences"][0]["tokens"];
		d_b = parsed["sentences"][0]["pos"];
		posdict = dict();
		for i in range(0, len(d_a)):
			posdict[d_a[i]] = d_b[i];
		for i in range(0, len(train_utterances[j])):
			train_utterances[j][i][2] = posdict.get(train_utterances[j][i][0], "NN");
		#print(train_utterances[j]);
	for j in range(0, len(test_utterances)):
		parsed = proc.parse_doc(test_sents[j]);
		d_a = parsed["sentences"][0]["tokens"];
		d_b = parsed["sentences"][0]["pos"];
		posdict = dict();
		for i in range(0, len(d_a)):
			posdict[d_a[i]] = d_b[i];
		for i in range(0, len(test_utterances[j])):
			test_utterances[j][i][2] = posdict.get(test_utterances[j][i][0], "NN");



	################# GET ACTUAL VECTORS
	X_train = [sent2features(s) for s in train_utterances];
	y_train = [sent2labels(s) for s in train_utterances];
	X_test = [sent2features(s) for s in test_utterances];
	y_test = [sent2labels(s) for s in test_utterances];
	# train the CRF model
	trainer = pycrfsuite.Trainer(verbose=False);
	for xseq, yseq in zip(X_train, y_train):
		trainer.append(xseq, yseq);
	# set model parameters
	trainer.set_params({
		"c1": 1.0,   # coefficient for L1 penalty
		"c2": 1e-3,  # coefficient for L2 penalty
		"max_iterations": 50,  # stop earlier
		# include transitions that are possible, but not observed
		"feature.possible_transitions": True
	});
	cprint("Training CRF...");
	trainer.train("open_taggers/crf_tagger" + str(fold));
	# evaluate the tagger
	tagger = pycrfsuite.Tagger();
	tagger.open("open_taggers/crf_tagger" + str(fold));
	y_pred = [tagger.tag(xseq) for xseq in X_test];

	ents_file = open("fold" + str(fold) + "_found_entities", "w");
	for utt in range(len(test_utterances)):
		print("TEST UTTERANCE: " + str(test_utterances[utt]));
		print("Y PRED: " + str(y_pred[utt]));
		entz = list();
		lent = None;
		for tokz in range(len(y_pred[utt])):
			#print(test_utterances[utt][tokz][0] + ":" + y_pred[utt][tokz][0]);
			#print("LENT: " + str(lent));
			#print("YPRED: " + str(y_pred[utt][tokz][0]));
			#if y_pred[utt][tokz][0] == "B":
			if y_pred[utt][tokz] == "B-ORGANIZATION" or y_pred[utt][tokz] == "B-PERSON":
				if lent != None:
					if y_pred[utt][tokz] == test_utterances[utt][tokz][1]:
						entz.append(lent);
					lent = None;
				lent = test_utterances[utt][tokz][0];
			#elif y_pred[utt][tokz][0] == "I":
			elif y_pred[utt][tokz] == "I-ORGANIZATION" or y_pred[utt][tokz] == "I-PERSON":
				#lent += " " + test_utterances[utt][tokz][0];
				pass;
			else:
				if lent != None:
					if y_pred[utt][tokz] == test_utterances[utt][tokz][1]:
						entz.append(lent);
					lent = None;
		if lent != None:
			if y_pred[utt][tokz] == test_utterances[utt][tokz][1]:
				entz.append(lent);
			lent = None;
		ents_file.write(str(entz) + "\n");
	ents_file.flush();
	ents_file.close();

def word2features(sent, i):
	word = sent[i][0];
	postag = sent[i][1];
	word1 = None;
	if i > 0:
		word1 = sent[i-1][0];
	features = [
		"bias",
		"word.lower=" + word.lower(),
		#"word[-3:]=" + word[-3:],
		#"word[-2:]=" + word[-2:],
		"word.isupper=%s" % word.isupper(),
		"word.istitle=%s" % word.istitle(),
		"word.isdigit=%s" % word.isdigit(),
		"word.length=%s" % len(word),
		"postag=" + postag,
		"postag[:2]=" + postag[:2],
	];
	if i > 0:
		word1 = sent[i-1][0];
		postag1 = sent[i-1][1];
		features.extend([
			"-1:word.lower=" + word1.lower(),
			"-1:word.istitle=%s" % word1.istitle(),
			"-1:word.isupper=%s" % word1.isupper(),
			"-1:word.isdigit=%s" % word1.isdigit(),
			"-1:word.length=%s" % len(word1),
			"-1:postag=" + postag1,
			"-1:postag[:2]=" + postag1[:2],
		]);
	else:
		features.append("BOS");
	if i < len(sent)-1:
		word1 = sent[i+1][0];
		postag1 = sent[i+1][1];
		features.extend([
			"+1:word.lower=" + word1.lower(),
			"+1:word.istitle=%s" % word1.istitle(),
			"+1:word.isupper=%s" % word1.isupper(),
			"+1:word.isdigit=%s" % word1.isdigit(),
			"+1:word.length=%s" % len(word1),
			"+1:postag=" + postag1,
			"+1:postag[:2]=" + postag1[:2],
		]);
	else:
		features.append('EOS');
	return features;

def isProfTitle(word):
	return word.lower() in dicts.getProfTitles();

def isPeopleWord(word):
	return word.lower() in dicts.getPeopleWords();

def isProfWord(word):
	return word.lower() in dicts.getProfWords();

def isEECSWord(word):
	return word.lower() in dicts.getEECSWords();

def sent2features(sent):
	reval = [word2features(sent, i) for i in range(len(sent))];
	return reval;

def sent2labels(sent):
	#print("\n\n\n" + str(sent));
	return [label for token, label, postag in sent];

def cprint(msg):
	tmsg = msg;
	st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S');
	tmsg = str(st) + ": " + str(tmsg);
	print(tmsg);
	log_file = open("open_crf.log", "a");
	log_file.write(tmsg + "\n");
	log_file.flush();
	log_file.close();

if __name__ == "__main__":
	main();
