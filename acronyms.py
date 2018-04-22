import math
import dicts

def main():
	#i have an acronym feature
	#how to do sequences? the previous word needs to be
	pword = "machine";
	word = "learning";
	word2 = "derping";
	print("machine learning: " + str(isSequence(pword, word)));
	print("machine derping: " + str(isSequence(pword, word2)));
	print("\n\n");
	pp = "to";
	print("to AI: " + str(isSequence(pp, "AI")));
	print("to arti: " + str(isSequence(pp, "arti")));
	print("to artiz: " + str(isSequence(pp, "artiz")));
	print("TA intel: " + str(isSequence("TA", "intel")));

def isAcronym(input):
	lexicon = dicts.getEECSdict().values();
	lexicon.extend(dicts.getEECSprofs());
	for i in range(0, len(lexicon)):
		parts = lexicon[i].lower().split(" ");
		if acronymInArray(parts, input):
			return True;
	return False;

def acronymInArray(parts, input):
	bmax = math.pow(2, len(parts));
	for j in range(1, int(bmax)):
		thetry = "";
		bstr = padString(str(bin(j))[2:], len(parts));
		for k in range(0, len(bstr)):
			if "1" == bstr[k]:
				thetry += parts[k][0];
		if thetry == input.lower():
			return True;
	return False;

def isSequence(pword, word):
	if pword == None:
		return False;
	lexicon = dicts.getEECSdict().values();
	lexicon.extend(dicts.getEECSprofs());
	for i in range(0, len(lexicon)):
		parts = lexicon[i].lower().split(" ");
		for j in range(0, len(parts) - 1):
			if parts[j].startswith(pword) and parts[j+1].startswith(word) or parts[j].startswith(pword) and acronymInArray(parts[j:], word) or acronymInArray(parts[:j+1], pword) and parts[j+1].startswith(word):
				return True;
	return False;

def padString(input, length):
	retstr = input;
	for i in range(0, length - len(input)):
		retstr = "0" + retstr;
	return retstr;

if __name__ == "__main__":
	main();
