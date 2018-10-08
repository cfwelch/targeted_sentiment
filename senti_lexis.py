
posw = list()
negw = list()
neuw = list()
negates = list()

def main():
	onLoad()
	string = "I absolutely love EECS but I hate everyone"
	print(lexCounts(string))
	print(lexNegate("won't"))
	print(lexNegate("negatory"))
	print(lexNegate("potato"))

def lexNegate(string):
	retval = False
	if string in negates:
		retval = True
	for k in negates:
		if k.endswith("*"):
			if string.startswith(k[:-1]):
				retval = True
	return retval

def lexCounts(string):
	pcount = 0
	ncount = 0
	nucount = 0
	for i in string.split():
		word = i.strip().lower()
		if word in posw:
			pcount += 1
		if word in negw:
			ncount += 1
		if word in neuw:
			nucount += 1
	return [pcount, ncount, nucount]

def onLoad():
	posF = open("./data/bing_liu_lexicon/pos_words")
	lines = posF.readlines()
	posF.close()
	for p in lines:
		posw.append(p.strip())
	negF = open("./data/bing_liu_lexicon/neg_words")
	lines = negF.readlines()
	negF.close()
	for n in lines:
		negw.append(n.strip())
	print("bing liu lexicon loaded")
	mpqaF = open("./data/mpqa/subjclueslen1-HLTEMNLP05.tff")
	lines = mpqaF.readlines()
	mpqaF.close()
	for m in lines:
		line = m.split()
		polarity = line[5][line[5].index("=")+1:]
		word = line[2][line[2].index("=")+1:]
		if polarity == "negative":
			negw.append(word)
		elif polarity == "neutral":
			neuw.append(word)
		elif polarity == "positive":
			posw.append(word)
	print("mpqa lexicon loaded")
	nF = open("./data/negate")
	lines = nF.readlines()
	nF.close()
	for n in lines:
		negates.append(n.strip())
	print("liwc negates loaded")

if __name__ == "__main__":
	main()

if __name__ != "__main__":
	onLoad()
