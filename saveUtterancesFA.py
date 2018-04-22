import sys

def main():
	fo = open("../data/extract_samples/EECS_annotated_samples", "r");
	lines = fo.readlines();
	mode = False;
	cString = "";
	for i in range(0, len(lines)):
		data = lines[i].strip();
		if data.startswith("<"):
			mode = True;
		if not mode and "" != data:
			cString += data + "\n";
		if data.endswith(">"):
			mode = False;
	fsave1 = open("allUtterances", "w");
	fsave1.write(cString);
	fsave1.flush();
	fsave1.close();

if __name__ == "__main__":
	main();
