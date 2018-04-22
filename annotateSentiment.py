
def main():
	fi = open("../data/extract_samples/EECS_annotated_samples");
	fo = open("sentimentAnnotations", "w");
	lines = fi.readlines();
	fi.close();
	lastSent = "";
	entList = list();
	lastEnt = list();
	for i in range(0, len(lines)):
		line = lines[i].strip();
		if line == "":
			if lastEnt:
				entList.append(lastEnt);
			print(lastSent);
			fo.write(lastSent + "\n");
			fo.flush();
			for ent in entList:
				string = "";
				for entparts in ent:
					if string != "":
						string += "\n";
					string += entparts;
				sentiment = raw_input("What is the sentiment expressed towards " + string + "?\n");
				if sentiment == "p":
					sentiment = "positive";
				elif sentiment == "n":
					sentiment = "negative";
				elif sentiment == "o":
					sentiment = "neutral";
				fo.write(string + "\nsentiment=" + sentiment + ">\n");
				fo.flush();
			fo.write("\n");
			fo.flush();
			lastSent = "";
			entList = list();
			lastEnt = list();
		elif lastSent == "":
			lastSent = line;
		elif line.startswith("<"):
			if lastEnt:
				entList.append(lastEnt);
				lastEnt = list();
			lastEnt.append(line);
		elif lastEnt:
			if line.startswith("department") or line.startswith("id") or line.startswith("name"):
				toadd = line;
				if toadd.endswith(">"):
					toadd = toadd[:-1];
				lastEnt.append(line);
	fo.close();

if __name__ == "__main__":
	main();
