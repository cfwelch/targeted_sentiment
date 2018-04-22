import NLU

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

	f2 = open("RNTN_sent");
	gendict = f2.readlines();
	f2.close();

	sdict = dict();
	slist = list();
	parent = None;
	lastpart = None;
	for i in gendict:
		if i.startswith(":"):
			parent = i[1:].strip();
			sdict[parent] = dict();
			slist.append(parent);
		elif is_number(i.strip()):
			sdict[parent][lastpart] = int(i.strip());
		else:
			lastpart = i.strip();
			sdict[parent][lastpart] = -1;

	print(len(tagset));
	print(len(sdict.keys()));
	print(len(sent_to_xtc));
	print(len(targets));

	tries = 0;
	correct = 0;
	for q in range(0, len(slist)):
		print(sdict[slist[q]]);
		print(sent_to_xtc[q]);
		for i in sent_to_xtc[q]:
			print(str(tagset[i]) + ":" + str(targets[i]));
			for j in sdict[slist[q]]:
				if tagset[i][0] in j:
					asent = "neutral";
					if int(sdict[slist[q]][j]) > 2:
						asent = "positive";
					elif int(sdict[slist[q]][j]) < 1:
						asent = "negative";
					print(asent);
					tries += 1;
					if targets[i] == asent:
						correct += 1;
	print("correct: " + str(correct*1.0/tries));

def is_number(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

if __name__ == "__main__":
	main();
