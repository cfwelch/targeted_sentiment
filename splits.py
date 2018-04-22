import random
import NLU

######################################################################################
# To stratify with 1/3 test split across:
#   1. Instructors                 +1
#   2. Classes                     +1
#   3. Positive, Negative, Neutral +3
#   4. BIOYZA                      +5
#
# I need to record 1/3 the total number of each in the data set, shuffle the set and
# then continue to add things as long as adding an utterance does not exceed any of
# these ten limits.
######################################################################################
def main():
	#get class types
	allines = NLU.getALines();
	allU = NLU.getUtterances(allines);
	#print(allU[1041]);
	#print(NLU.getSlots(allU[1041]));
	#limits for test set
	poslim = 114;
	neglim = 80;
	neulim = 217;
	classlim = 325;
	instrlim = 85;
	blim = 85;
	ilim = 120;
	ylim = 85;
	zlim = 33;
	alim = 379;
	#store NER labels
	fi = open("CRFNERADVISE-BIOYZ", "r");#BIO
	nerLines = fi.readlines();
	fi.close();
	cur_sent = list();
	ner_sents = list();
	for i in range(0, len(nerLines)):
		if "" != nerLines[i][:-1]:
			cur_sent.append(tuple(nerLines[i][:-1].split("\t")));
		else:
			if len(cur_sent) > 0:
				ner_sents.append(cur_sent);
			cur_sent = list();

	#THIS CODE WILL PRINT THE NUMBER OF EACH TYPE OF TOKEN
	tok_b = 0;
	tok_i = 0;
	tok_o = 0;
	tok_y = 0;
	tok_z = 0;
	tok_a = 0;
	tok_q = 0;
	for i in range(0, len(ner_sents)):
		for j in range(0, len(ner_sents[i])):
			tok = ner_sents[i][j][2];
			if tok == "B":
				tok_b += 1;
			elif tok =="I":
				tok_i += 1;
			elif tok == "O":
				tok_o += 1;
			elif tok == "Y":
				tok_y += 1;
			elif tok == "Z":
				tok_z += 1;
			elif tok == "A":
				tok_a += 1;
			else:
				print ner_sents[i];
				tok_q += 1;
	print("B: " + str(tok_b) + " - " + str(tok_b*1.0/3));
	print("I: " + str(tok_i) + " - " + str(tok_i*1.0/3));
	print("O: " + str(tok_o) + " - " + str(tok_o*1.0/3));
	print("Y: " + str(tok_y) + " - " + str(tok_y*1.0/3));
	print("Z: " + str(tok_z) + " - " + str(tok_z*1.0/3));
	print("A: " + str(tok_a) + " - " + str(tok_a*1.0/3));
	print("Q: " + str(tok_q) + " - " + str(tok_q*1.0/3));

	#store sentiment labels
	fo = open("EECS_annotated_samples", "r");
	lines = fo.readlines();
	fo.close();
	mode = False;
	sents = list();
	targets = list();
	lastTaken = "";
	lastSent = "";
	isclass = False;
	tagset = list();
	lastTagset = list();
	index = 0;
	sent_to_xtc = dict();
	sent_to_xtc[0] = list();
	#I left in these data structures in case I am ever interested in what the distributions of individual splits look like.
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
			isclass = False;
			sents.append(lastSent);
			tagset.append(lastTagset);
			sent_to_xtc[index].append(len(sents)-1);
			if lastTaken == "":
				targets.append("neutral");
			else:
				targets.append(lastTaken);

	#some tests
	#print(len(sent_to_xtc));
	#print(sent_to_xtc[746]);
	#print(len(targets));
	#print(len(sents));
	#print(sents[sent_to_xtc[746][0]]);
	#print(sent_to_xtc[746]);
	#print(targets[sent_to_xtc[746][0]]);
	#print(targets[sent_to_xtc[746][1]]);
	#print(ner_sents[746]);

	#generate splits
	Nsplits = 100;
	fsavef = open("splits", "w");
	for i in range(0, Nsplits):
		testdata = range(0, index);
		random.shuffle(testdata);
		#counters
		poshas = 0;
		neghas = 0;
		neuhas = 0;
		classhas = 0;
		instrhas = 0;
		bhas = 0;
		ihas = 0;
		yhas = 0;
		zhas = 0;
		ahas = 0;
		print("i = " + str(i));
		nthsplitTR = list();
		nthsplitTE = list();
		for j in range(0, index):
			#counters
			poscur = 0;
			negcur = 0;
			neucur = 0;
			classcur = 0;
			instrcur = 0;
			bcur = 0;
			icur = 0;
			ycur = 0;
			zcur = 0;
			acur = 0;
			nslots = NLU.getSlots(allU[testdata[j]]);
			instrcur = len(nslots[1]);
			classcur = len(nslots[0]);
			#print("instructors: " + str(len(nslots[1])));
			#print("classes: " + str(len(nslots[0])));
			#print(testdata[j]);
			#print(sent_to_xtc[testdata[j]]);
			for k in range(0, len(sent_to_xtc[testdata[j]])):
				if targets[sent_to_xtc[testdata[j]][k]] == "positive":
					poscur += 1;
				elif targets[sent_to_xtc[testdata[j]][k]] == "negative":
					negcur += 1;
				elif targets[sent_to_xtc[testdata[j]][k]] == "neutral":
					neucur += 1;
			#print("poscur: " + str(poscur));
			#print("negcur: " + str(negcur));
			#print("neucur: " + str(neucur));
			#print(ner_sents[testdata[j]]);
			for k in range(0, len(ner_sents[testdata[j]])):
				if ner_sents[testdata[j]][k][2] == "B":
					bcur += 1;
				elif ner_sents[testdata[j]][k][2] == "I":
					icur += 1;
				elif ner_sents[testdata[j]][k][2] == "Y":
					ycur += 1;
				elif ner_sents[testdata[j]][k][2] == "Z":
					zcur += 1;
				elif ner_sents[testdata[j]][k][2] == "A":
					acur += 1;
			#print("bcur: " + str(bcur));
			#print("icur: " + str(icur));
			#print("ycur: " + str(ycur));
			#print("zcur: " + str(zcur));
			#print("acur: " + str(acur));
			if poscur + poshas < poslim and negcur + neghas < neglim and neucur + neuhas < neulim and bcur + bhas < blim \
			and icur + ihas < ilim and ycur + yhas < ylim and zcur + zhas < zlim and acur + ahas < alim and classcur + classhas < classlim \
			and instrcur + instrhas < instrlim:
				nthsplitTE.append(testdata[j]);
				poshas += poscur;
				neghas += negcur;
				neuhas += neucur;
				bhas += bcur;
				ihas += icur;
				yhas += ycur;
				zhas += zcur;
				ahas += acur;
				classhas += classcur;
			else:
				nthsplitTR.append(testdata[j]);
		print("TRAIN INSTANCES: " + str(len(nthsplitTR)));
		print("TEST INSTANCES: " + str(len(nthsplitTE)));
		fsavef.write(str(nthsplitTR) + ":" + str(nthsplitTE) + "\n");
		print("split(" + str(i) + "): " + str(nthsplitTR) + ":" + str(nthsplitTE));
	fsavef.flush();
	fsavef.close();

if __name__ == "__main__":
	main();
