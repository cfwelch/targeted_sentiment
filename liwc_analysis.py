import re

liwc_pairs = list();
liwc_dict = dict();

# this code generates parallel list
"""# read LIWC and generate liwc_pairs
fo = open("../data/LIWC.all", "r");
lines = fo.readlines();
fo.close();
for i in range(0, len(lines)):
	parts = lines[i].split(" ,");
	parts[0] = parts[0].replace("*", ".*");
	liwc_pairs.append((parts[0], parts[1].strip()));
# read BIOYZ and iterate over lines
fg = open("CRFNERADVISE-BIOYZ", "r");
linez = fg.readlines();
fg.close();
fk = open("LIWC-PARALLEL-NERBIOYZ", "w");
for i in range(0, len(linez)):
	if linez[i].strip() == "":
		fk.write("\n");
	else:
		theword = linez[i].split("\t")[0].lower();
		print(theword);
		fk.write(getLIWCWord(theword) + "\n");
fk.flush();
fk.close();"""
# need to write new stuff to compare them and order by frequency
def main():
	fg = open("CRFNERADVISE-BIOYZ", "r");
	linez = fg.readlines();
	fg.close();
	ff = open("LIWC-PARALLEL-NERBIOYZ", "r");
	lines = ff.readlines();
	ff.close();
	for i in range(0, len(linez)):
		if linez[i].strip() != "":
			parts = linez[i].strip().split("\t");
			#print(parts);
			partsNext = ["", "", ""];
			if i < len(linez) and linez[i+1].strip() != "":
				partsNext = linez[i+1].strip().split("\t");
			#print("partsNext: " + str(partsNext));
			#print(str(i) + ":" + str(len(linez)));
			if parts[2] == "B" or parts[2] == "I" or partsNext[2] == "B" or partsNext[2] == "I":
				print(parts[0] + "\t" + lines[i].strip() + "\t" + parts[2].strip());
			else:
				print("\n");

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

if __name__ == "__main__":
	main();
