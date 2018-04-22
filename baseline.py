import sys
import numpy
import re

def main():
	# initalize program
	fo = open("../data/extract_samples/EECS_annotated_samples_anonymized", "r");
	lines = fo.readlines();
	mode = False;
	targets = list();
	lastID = "";
	lastSent = "";
	isclass = False;
	correct = 0;
	incorrect = 0;
	TSE = 0; # total slots expected
	# parse annotated EECS file
	for i in range(len(lines)):
		data = lines[i].strip();
		if data.startswith("<class") or data.startswith("<instructor"):
			mode = True;
			lastID = "";
		if data.startswith("<class"):
			isclass = True;
		if mode and data.startswith("id="):
			lastID = data[3:];
			if lastID.endswith(">"):
				lastID = lastID[:-1];
		if not mode and "" != data:
			lastSent = data;
		if data == "":
			# test the REGEX labeler
			# print("targets for: " + lastSent);
			pattern = re.compile("\d{3}");
			matches = pattern.findall(lastSent);
			for j in range(len(targets)):
				# print("target: " + targets[j]);
				TSE += 1;
				if targets[j] not in matches:
					incorrect += 1;
					print("Could not find " + targets[j] + " in " + lastSent);
			for j in range(len(matches)):
				if matches[j] in targets:
					correct += 1;
				else:
					incorrect += 1;
					print("Found false positive " + matches[j] + " in " + lastSent);
			# update evaluation variables
			del targets[:];
		if data.endswith(">"):
			mode = False;
			if isclass:
				isclass = False;
				if lastID != "":
					targets.append(lastID);

	precision = correct * 1.0 / (correct + incorrect);
	recall = correct * 1.0 / TSE;
	print("Classifier Precision: " + str(precision));
	print("Classifier Recall: " + str(recall));

if __name__ == "__main__":
	main();
