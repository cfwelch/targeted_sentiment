
import NLU

def main():
	fi = open("sentimentAnnotations");
	line1 = fi.readlines();
	fi.close();
	fo = open("EECS_annotated_samples_anonymized");
	line2 = fo.readlines();
	fo.close();
	utt1 = NLU.getUtterances(line1);
	utt2 = NLU.getUtterances(line2);
	correct = 0;
	wrong = 0;
	NEU_NEG = 0;
	NEU_POS = 0;
	POS_NEG = 0;
	SNEU_NEG = set();
	SNEU_NEG.add("neutral");
	SNEU_NEG.add("negative");
	SNEU_POS = set();
	SNEU_POS.add("neutral");
	SNEU_POS.add("positive");
	SPOS_NEG = set();
	SPOS_NEG.add("negative");
	SPOS_NEG.add("positive");
	disagrees = list();
	inst = 1;
	insttype = "neutral";
	for i in range(0, len(utt1)):
		slots1 = NLU.getSlots(utt1[i]);
		slots2 = NLU.getSlots(utt2[i]);
		for j in range(0, len(slots1[0])):
			if insttype == slots2[0][j][2]:
				inst += 1;
			if slots1[0][j][3] == slots2[0][j][3]:
				correct += 1;
			else:
				tset = set();
				tset.add(slots1[0][j][3]);
				tset.add(slots2[0][j][3]);
				disagrees.append(utt1[i])
				if slots2[0][j][3] == insttype:
					if tset == SNEU_NEG:
						NEU_NEG += 1;
					elif tset == SNEU_POS:
						NEU_POS += 1;
					elif tset == SPOS_NEG:
						POS_NEG += 1;
				wrong += 1;
		for j in range(0, len(slots1[1])):
			if slots1[1][j][1] == slots2[1][j][1]:
				correct += 1;
			else:
				tset = set();
				disagrees.append(utt1[i])
				tset.add(slots1[1][j][1]);
				tset.add(slots2[1][j][1]);
				if slots2[1][j][1] == insttype:
					if tset == SNEU_NEG:
						NEU_NEG += 1;
					elif tset == SNEU_POS:
						NEU_POS += 1;
					elif tset == SPOS_NEG:
						POS_NEG += 1;
				wrong += 1;
	print("Agree on " + str(correct));
	print("Disagree on " + str(wrong));
	print("Percent agreement is " + str(correct*1.0/(correct+wrong)) + "%");
	#print("NEU_NEG: " + str(NEU_NEG*1.0/(correct+wrong)));
	#print("NEU_POS: " + str(NEU_POS*1.0/(correct+wrong)));
	#print("POS_NEG: " + str(POS_NEG*1.0/(correct+wrong)));
	print("NEU_NEG: " + str(NEU_NEG*1.0/inst));
	print("NEU_POS: " + str(NEU_POS*1.0/inst));
	print("POS_NEG: " + str(POS_NEG*1.0/inst));
	#for i in disagrees:
	#	print(i[0]);

if __name__ == "__main__":
	main();
