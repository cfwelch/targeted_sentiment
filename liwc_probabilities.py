import operator

def main():
	ff = open("LIWC-CORRELATION-OUTPUT", "r");
	lines = ff.readlines();
	ff.close();
	tlist = list();
	current = list();
	for i in range(0, len(lines)):
		if lines[i].strip() != "":
			current.append(lines[i].strip().split("\t")[1]);
		else:
			if current:
				tlist.append(current);
				current = list();
	print("TLIST: " + str(tlist));
	thed = dict();
	for i in range(0, len(tlist)):
		for j in range(0, len(tlist[i]) - 1):
			pat = tlist[i][j] + "-" + tlist[i][j+1];
			if pat in thed:
				thed[pat] += 1;
			else:
				thed[pat] = 1;
	newd = sorted(thed.items(), key=operator.itemgetter(1))
	for key in newd:
		print(key);

if __name__ == "__main__":
	main();
