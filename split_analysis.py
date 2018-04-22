import leastSquares
import NLU

def main():
	# get scores
	fscores = open("S1feature");#S1feature -S1single_lies
	lines = fscores.readlines();
	fscores.close();
	scores = list();
	for i in lines:
		scores.append(float(i.strip()));
	sort_scores = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1])];
	sort_scores.reverse();

	# get splits
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

	#get speakers
	nlines = NLU.getALines();
	utterances = NLU.getUtterances(nlines);
	nlist = list();
	for i in range(0, len(splits)):
		senti_utters = list();
		for j in range(0, len(splits[i][0])):
			senti_utters.append(utterances[splits[i][0][j]]);
		likesMatrix, slist = leastSquares.getMatrix(senti_utters);

		test_utters = list();
		for j in range(0, len(splits[i][1])):
			test_utters.append(utterances[splits[i][1][j]]);
		TlikesMatrix, Tslist = leastSquares.getMatrix(test_utters);

		nonneus = 0;
		nnews = 0;
		density = 0.0;
		counts = list();
		#iterate over rows
		for k in range(0, len(likesMatrix)):
			nonneu = 0;
			for j in range(0, len(likesMatrix[k])):
				if int(likesMatrix[k][j]) != 5:
					nonneu += 1;
			if nonneu > 0:
				nnews += 1;
			nonneus += nonneu;
			counts.append(nonneu);
		#iterate over columns
		elaps = 0;
		for k in range(0, len(likesMatrix[0])):
			nonneu = 0;
			TNEW = 0;
			for j in range(0, len(likesMatrix)):
				if int(likesMatrix[j][k]) != 5:
					nonneu = 1;
				if int(TlikesMatrix[j][k]) != 5:
					TNEW = 1;
			if nonneu == 1 and TNEW == 1:
				elaps += 1;

		nlist.append(str(nnews) + ":" + str(nonneus) + ":" + str(counts) + ":" + str(elaps));

	#print correlations
	for i in sort_scores:
		print(str(scores[i]) + " - " + nlist[i]);


if __name__ == "__main__":
	main();
