from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
import sklearn
import dicts
import numpy
#import sentiment_treemaker
import entity_distance
import pycrfsuite
import re
import acronyms

liwc_pairs = list();
liwc_dict = dict();

# TODO: this file is a mess... is there a way I can clean this up?
def main():
	bioset = ['A', 'B', 'I', 'O', 'Y', 'Z'];
	fo = open("CRFNERADVISE-BIOYZ", "r");#BIO
	lines = fo.readlines();
	fi = open("splits", "r");
	splitsin = fi.readlines();
	fo.close();
	fi.close();
	# load LIWC
	#fg = open("../data/LIWC.all", "r");
	#lines2 = fg.readlines();
	#fg.close();
	#for i in range(0, len(lines2)):
	#	parts = lines2[i].split(" ,");
	#	parts[0] = parts[0].replace("*", ".*");
	#	liwc_pairs.append((parts[0], parts[1]));
	# format splits
	fsplits = open("splits");
	splines = fsplits.readlines();
	splits = list();
	for i in range(0, len(splines)):
		parts = splines[i].strip().split(":");
		train = list();
		test = list();
		for s in parts[0][1:-1].split(", "):
			train.append(int(s));
		for s in parts[1][1:-1].split(", "):
			test.append(int(s));
		splits.append((train, test));
	fsplits.close();
	Nsplits = len(splits);
	print("Number of Splits: " + str(Nsplits));
	# for each split
	recall = [0] * len(bioset);
	precision = [0] * len(bioset);
	fscore = [0] * len(bioset);
	support = [0] * len(bioset);
	fscores = open("fscores", "w");
	rscores = open("rscores", "w");
	pscores = open("pscores", "w");
	for split in range(0, Nsplits):
		print("Calculating split " + str(split + 1) + " ...");
		# separate sentences into train/test
		cur_sent = list();
		train_sents = list();
		test_sents = list();
		sent_index = 0;
		for i in range(0, len(lines)):
			if "" != lines[i][:-1]:
				cur_sent.append(tuple(lines[i][:-1].split("\t")));
			else:
				if len(cur_sent) > 0:
					if sent_index in splits[split][1]:
						test_sents.append([sent_index, cur_sent]);
					else:
						train_sents.append([sent_index, cur_sent]);
					sent_index += 1;
				cur_sent = list();
		print("Train Length: " + str(len(train_sents)));
		print("Test Length: " + str(len(test_sents)));
		# test sentence feature extraction
		#print(sent2features(train_sents[0])[0]);
		#create topic features
		#LDA is currently turned OFF
		#vocab, topic_words = LDAlda.getTopics(train_sents);
		vocab = None;
		topic_words = None;
		X_train = [sent2features(s, vocab, topic_words) for s in train_sents];
		y_train = [sent2labels(s) for s in train_sents];
		X_test = [sent2features(s, vocab, topic_words) for s in test_sents];
		y_test = [sent2labels(s) for s in test_sents];
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
		trainer.train("taggers/advising_crf_tagger" + str(split));
		# evaluate the tagger
		tagger = pycrfsuite.Tagger();
		tagger.open("taggers/advising_crf_tagger" + str(split));
		y_pred = [tagger.tag(xseq) for xseq in X_test];
		#print("length y_pred: " + str(len(y_pred)));
		print("y_pred[0]: " + str(y_pred[0]));
		#print("test_sents[0]: " + str(test_sents[2]));
		#outs = bio_classification_report(y_test, y_pred);
		#print(outs);
		evouts = evaluation(y_test, y_pred, bioset);
		print("evouts length: " + str(len(evouts)));
		cindz = evouts[1];
		evouts = evouts[0];
		print(evouts);
		#print("L" + str(len(evouts)));
		#print("L" + str(len(evouts[0])));
		#print("ytest: " + str(y_test));
		#print("ypred: " + str(y_pred));
		#recall - precision - fscore
		print(evouts[2]);
		fscores.write(str(evouts[2]) + "\n");
		pscores.write(str(evouts[0]) + "\n");
		rscores.write(str(evouts[1]) + "\n");
		#for i in range(0, len(evouts[0])):
			#tokenmap.put("EECS488", "A");print(str(i));
			#precision[i] += evouts[0][i] * evouts[3][i];
		#for i in range(0, len(evouts[1])):
			#recall[i] += evouts[1][i] * evouts[3][i];
		#for i in range(0, len(evouts[2])):
			#fscore[i] += evouts[2][i] * evouts[3][i];
		#for i in range(0, len(evouts[3])):
			#support[i] += evouts[3][i];
	fscores.flush();
	fscores.close();
	pscores.flush();
	pscores.close();
	rscores.flush();
	rscores.close();
	#print("& precision & recall & fscore & support");
	#for i in range(0, len(bioset)):
		#cindz can be used to get the right class name somehow...
		#print(bioset[i] + " & " + str(precision[i] / support[i]) + " & " + str(recall[i] / support[i]) + " & " + str(fscore[i] / support[i]) + " & " + str(support[i]));

def word2features(sent, i, treemodel, vocab, topic_words):
	word = sent[i][0];
	postag = sent[i][1];
	word1 = None;
	if i > 0:
		word1 = sent[i-1][0];
	#tsent = sentiment_treemaker.lookup(word, treemodel);
	features = [
		"bias",
		"word.lower=" + word.lower(),
		#"word[-3:]=" + word[-3:],
		#"word[-2:]=" + word[-2:],
		"word.isupper=%s" % word.isupper(),
		"word.istitle=%s" % word.istitle(),
		"word.isdigit=%s" % word.isdigit(),
		"word.isprofword=%s" % isProfWord(word),
		"word.iseecsword=%s" % isEECSWord(word),
		#"word.ispeopleword=%s" % isPeopleWord(word),
		"word.length=%s" % len(word),
		"word.isacronym=%s" % acronyms.isAcronym(word),
		"word.issequence=%s" % acronyms.isSequence(word1, word),
		#"word.liwcword=%s" % getLIWCWord(word),
		"word.isproftitle=%s" % isProfTitle(word),
		"word.nearestEntity=%s" % entity_distance.nearestEntity(word),
		#"word.ldatopic=%s" % LDAlda.getTopic(word, vocab, topic_words),
		#"word.treesentiment12=%s" % str(tsent[0]) + ":" + str(tsent[1]),
		#"word.treesentiment23=%s" % str(tsent[1]) + ":" + str(tsent[2]),
		#"word.treesentiment2=%s" % str(tsent[1]),
		#"word.treesentiment3=%s" % str(tsent[2]),
		"postag=" + postag,
		"postag[:2]=" + postag[:2],
	];
	if i > 0:
		word1 = sent[i-1][0];
		postag1 = sent[i-1][1];
		#tsent1 = sentiment_treemaker.lookup(word1, treemodel);
		features.extend([
			"-1:word.lower=" + word1.lower(),
			"-1:word.istitle=%s" % word1.istitle(),
			"-1:word.isupper=%s" % word1.isupper(),
			"-1:word.isdigit=%s" % word1.isdigit(),
			"-1:word.isprofword=%s" % isProfWord(word1),
			"-1:word.iseecsword=%s" % isEECSWord(word1),
			#"-1:word.ispeopleword=%s" % isPeopleWord(word1),
			"-1:word.length=%s" % len(word1),
			#"-1:word.liwcword=%s" % getLIWCWord(word1),
			"-1:word.isproftitle=%s" % isProfTitle(word1),
			"-1:word.nearestEntity=%s" % entity_distance.nearestEntity(word1),
			#"-1:word.ldatopic=%s" % LDAlda.getTopic(word1, vocab, topic_words),
			#"-1:word.treesentiment12=%s" % str(tsent1[0]) + ":" + str(tsent1[1]),
			#"-1:word.treesentiment23=%s" % str(tsent1[1]) + ":" + str(tsent1[2]),
			#"-1:word.treesentiment2=%s" % str(tsent1[1]),
			#"-1:word.treesentiment3=%s" % str(tsent1[2]),
			"-1:postag=" + postag1,
			"-1:postag[:2]=" + postag1[:2],
		]);
	else:
		features.append("BOS");
	'''if i > 1:
		word2 = sent[i-2][0];
		postag2 = sent[i-2][1];
		features.extend([
			"-2:word.lower=" + word2.lower(),
			"-2:word.istitle=%s" % word2.istitle(),
			"-2:word.isupper=%s" % word2.isupper(),
			"-2:postag=" + postag2,
			"-2:postag[:2]=" + postag2[:2],
		]);
	else:
		features.append("BBOS");'''
	if i < len(sent)-1:
		word1 = sent[i+1][0];
		postag1 = sent[i+1][1];
		#tsent1 = sentiment_treemaker.lookup(word1, treemodel);
		features.extend([
			"+1:word.lower=" + word1.lower(),
			"+1:word.istitle=%s" % word1.istitle(),
			"+1:word.isupper=%s" % word1.isupper(),
			"+1:word.isdigit=%s" % word1.isdigit(),
			"+1:word.isprofword=%s" % isProfWord(word1),
			"+1:word.iseecsword=%s" % isEECSWord(word1),
			#"+1:word.ispeopleword=%s" % isPeopleWord(word1),
			"+1:word.length=%s" % len(word1),
			#"+1:word.liwcword=%s" % getLIWCWord(word1),
			"+1:word.isproftitle=%s" % isProfTitle(word1),
			"+1:word.nearestEntity=%s" % entity_distance.nearestEntity(word1),
			#"+1:word.ldatopic=%s" % LDAlda.getTopic(word1, vocab, topic_words),
			#"+1:word.treesentiment12=%s" % str(tsent1[0]) + ":" + str(tsent1[1]),
			#"+1:word.treesentiment23=%s" % str(tsent1[1]) + ":" + str(tsent1[2]),
			#"+1:word.treesentiment2=%s" % str(tsent1[1]),
			#"+1:word.treesentiment3=%s" % str(tsent1[2]),
			"+1:postag=" + postag1,
			"+1:postag[:2]=" + postag1[:2],
		]);
	else:
		features.append('EOS');
	'''if i < len(sent)-2:
		word2 = sent[i+2][0];
		postag2 = sent[i+2][1];
		features.extend([
			'+2:word.lower=' + word2.lower(),
			'+2:word.istitle=%s' % word2.istitle(),
			'+2:word.isupper=%s' % word2.isupper(),
			'+2:postag=' + postag2,
			'+2:postag[:2]=' + postag2[:2],
		]);
	else:
		features.append('EEOS');'''
	return features;

def getLIWCWord(word):
	found = "NONE";
	prev = False;
	# LIWC regex cache
	if word in liwc_dict:
		found = liwc_dict[word];
		prev = True;
	else:
		for i in range(0, len(liwc_pairs)):
			#use anchors ^whatever$
			if re.match(liwc_pairs[i][0], word):
				#print("matches " + liwc_pairs[i][0] + ":" + liwc_pairs[i][1]);
				found = liwc_pairs[i][1];
				break;
	if not prev:
		liwc_dict[word] = found;
	return found;

def isProfTitle(word):
	return word.lower() in dicts.getProfTitles();

def isPeopleWord(word):
	return word.lower() in dicts.getPeopleWords();

def isProfWord(word):
	return word.lower() in dicts.getProfWords();

def isEECSWord(word):
	return word.lower() in dicts.getEECSWords();

def sent2features(sent, vocab, topic_words):
	#print("Current Sentence: " + str(sent[0]));
	#treemodel = sentiment_treemaker.getListModel(sent[0]);
	treemodel = 0;
	reval = [word2features(sent[1], i, treemodel, vocab, topic_words) for i in range(len(sent[1]))];
	#print(sent[1][0][0]);
	#if sent[1][0][0] == "Linjia":
	#	print(reval);
	return reval;

def sent2featuresWithSent(sent, strsent):
	#treemodel = sentiment_treemaker.getMappedModel(strsent.strip());
	treemodel = 0;
	return [word2features(sent, i, treemodel, 0, 0) for i in range(len(sent))];

def sent2labels(sent):
	return [label for token, postag, label in sent[1]];

def sent2tokens(sent):
	return [token for token, postag, label in sent];

def evaluation(y_true, y_pred, bioset):
	lb = LabelBinarizer();
	y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)));
	y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)));

	#tagset = set(lb.classes_);# - {'O'};
	tagset = set(bioset);
	#print("Tset: " + str(Tset));
	tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1]);
	class_indices = {cls: idx for idx, cls in enumerate(bioset)};#lb.classes_
	print("CLASS INDICIES: " + str(class_indices));
	#print([class_indices[cls] for cls in tagset]);
	#for cls in tagset:
	#	print(cls + ":" + str(class_indices[cls]));

	return [precision_recall_fscore_support(
		y_true_combined,
		y_pred_combined,
		labels = [class_indices[cls] for cls in tagset],
		average = None,
		sample_weight = None,
		#target_names = tagset,
	), class_indices];

def bio_classification_report(y_true, y_pred):
	"""
	Classification report for a list of BIO-encoded sequences.
	It computes token-level metrics and discards "O" labels.
	
	Note that it requires scikit-learn 0.15+ (or a version from github master)
	to calculate averages properly!
	"""
	lb = LabelBinarizer();
	y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)));
	y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)));

	tagset = set(lb.classes_);# - {'O'};
	tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1]);
	class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)};

	return classification_report(
		y_true_combined,
		y_pred_combined,
		labels = [class_indices[cls] for cls in tagset],
		target_names = tagset,
	);

if __name__ == "__main__":
	main();
