import NLU

s_map = dict();
s_set = set();
likes_map = dict();
sim_likes = dict();

# intended control flow:
# 1. determine speaker     - in: utterance             - out: speaker
# 2. find similar speakers - in: speaker               - out: speakers
# 3. find entities         - in: utterance             - out: entities
# 4. add up speaker sents  - in: speaker and entity    - out: number
def main():
	name = "MEGHAN";
	fi = open("../data/extract_samples/pID_AEU");
	pid = fi.readlines();
	fi.close();
	pidmap = dict();
	pset = set();
	for i in range(0, len(pid)):
		parts = pid[i].split("\t");
		pset.add(parts[0]);
		pidmap[parts[1].strip()] = parts[0];
	fl = open("../data/extract_samples/EECS_annotated_samples");
	lines = fl.readlines();
	fl.close();
	utterances = NLU.getUtterances(lines);
	print(utterances[0]);
	print("Speaker: " + pidmap[utterances[0][0].strip()]);
	slots = NLU.getSlots(utterances[0]);
	print(slots);
	plikes = dict();
	for i in pset:
		plikes[i] = [list(), list()];
	for i in range(0, len(utterances)):
		slots = NLU.getSlots(utterances[i]);
		speaker = pidmap[utterances[i][0].strip()];
		if slots[0]:
			plikes[speaker][0].extend(slots[0]);
		if slots[1]:
			plikes[speaker][1].extend(slots[1]);
	print("\n\nGiven that EECS 492 sentiment is neutral...");
	#print(plikes[name]);
	wholikes = ("EECS","492","neutral");
	likers = list();
	for i in pset:
		if wholikes in plikes[i][0]:
			likers.append(i);
	# check instructors in likers
	ucontains_i = "Quentin Stout";
	print("\n\nWho likes " + ucontains_i);
	for i in likers:
		for j in range(0, len(plikes[i][1])):
			if plikes[i][1][j][0] == ucontains_i:
				print(i + ": " + str(plikes[i][1][j]));
	# check classes in likers
	ucontains_cd = "EECS";
	ucontains_cid = "545";
	print("\n\nWho likes " + ucontains_cd + " " + ucontains_cid);
	for i in likers:
		for j in range(0, len(plikes[i][0])):
			# don't worry about department but if you want to... then use this line
			# plikes[i][0][j][0] == ucontains_cd and 
			if plikes[i][0][j][1] == ucontains_cid:
				print(i + ": " + str(plikes[i][0][j]));
	# find all people with similar sentiments to <name> in the data set
	print("\n\nSimlikes!");
	simlikesmap = dict();
	for q in pset:
		simlikes = list();
		for i in pset:
			if i == q:
				continue;
			found = False;
			for j in range(0, len(plikes[i][0])):
				if (("EECS",plikes[i][0][j][1],plikes[i][0][j][2]) in plikes[name][0] or ("",plikes[i][0][j][1],plikes[i][0][j][2]) in plikes[name][0]) and plikes[i][0][j][2] != "neutral":
					print("similar likes for " + i + " and " + name + ": " + str(plikes[i][0][j]));
					simlikes.append(i);
					found = True;
					break;
			if not found:
				for j in range(0, len(plikes[i][1])):
					if plikes[i][1][j] in plikes[name][1] and plikes[i][1][j][1] != "neutral":
						print("similar likes for " + i + " and " + name + ": " + str(plikes[i][1][j]));
						simlikes.append(i);
						found = True;
						break;
		simlikesmap[q] = simlikes;
	# calculate % of times where OSCORE will be nonzero
	times = 0;
	ttimes = 0;
	for u in utterances:
		slots = NLU.getSlots(u);
		speaker = pidmap[u[0].strip()];
		for slot in slots[0]:
			ttimes += 1;
			oscore = 0;
			for i in simlikesmap[speaker]:
				pscore = 0;
				for j in range(0, len(plikes[i][0])):
					if slot[1] == plikes[i][0][j][1]:
						if plikes[i][0][j][2] == "positive":
							pscore += 1;
						elif plikes[i][0][j][2] == "negative":
							pscore -= 1;
				if pscore > 0:
					oscore += 1;
				elif pscore < 0:
					oscore -= 1;
			if oscore != 0:
				times += 1;
		for slot in slots[1]:
			ttimes += 1;
			oscore = 0;
			for i in simlikesmap[speaker]:
				pscore = 0;
				for j in range(0, len(plikes[i][1])):
					if slot[0] == plikes[i][1][j][0]:
						if plikes[i][1][j][1] == "positive":
							pscore += 1;
						elif plikes[i][1][j][1] == "negative":
							pscore -= 1;
				if pscore > 0:
					oscore += 1;
				elif pscore < 0:
					oscore -= 1;
			if oscore != 0:
				times += 1;
	print("Times: " + str(times));
	print("Total Times: " + str(ttimes));
	print("Percentage: " + str(times*100.0/ttimes));

# takes speaker name and class entity
def getSelfScore(speaker, entity):
	sscore = 0;
	for i in range(0, len(likes_map[speaker][0])):
		if entity == likes_map[speaker][0][i][1]:
			if likes_map[speaker][0][i][2] == "positive":
				sscore += 1;
			elif likes_map[speaker][0][i][2] == "negative":
				sscore -= 1;
	return sscore;

# takes in speaker and entity
def getOtherScore(name, entity):
	oscorePOS = 0;
	oscoreNEG = 0;
	for i in sim_likes[name]:
		pscore = 0;
		for j in range(0, len(likes_map[i][0])):
			if entity == likes_map[i][0][j][1]:
				if likes_map[i][0][j][2] == "positive":
					#print(i + " also likes " + str(entity));
					pscore += 1;
				elif likes_map[i][0][j][2] == "negative":
					#print(i + " also dislikes " + str(entity));
					pscore -= 1;
		for j in range(0, len(likes_map[i][1])):
			if entity == likes_map[i][1][j][0]:
				if likes_map[i][1][j][1] == "positive":
					pscore += 1;
				elif likes_map[i][1][j][1] == "negative":
					pscore -= 1;
		if pscore > 0:
			oscorePOS += 1;
		elif pscore < 0:
			oscoreNEG += 1;
	return [oscorePOS, oscoreNEG];

def getSpeaker(utterance):
	return s_map[utterance.strip()];

# Must be called before other functions.
# Utterance input is list of lists of lines from annotation file.
def genLikesMap(utterances):
	likes_map.clear();
	for i in s_set:
		likes_map[i] = [list(), list()];
	for i in range(0, len(utterances)):
		slots = NLU.getSlots(utterances[i]);
		speaker = s_map[utterances[i][0].strip()];
		if slots[0]:
			likes_map[speaker][0].extend(slots[0]);
		if slots[1]:
			likes_map[speaker][1].extend(slots[1]);
	# generate dictionary for similar likes for each person
	for q in s_set:
		simlikeq = list();
		for i in s_set:
			if i == q:
				continue;
			found = False;
			for j in range(0, len(likes_map[i][0])):
				if (("EECS",likes_map[i][0][j][1],likes_map[i][0][j][2]) in likes_map[q][0] or ("",likes_map[i][0][j][1],likes_map[i][0][j][2]) in likes_map[q][0]) and likes_map[i][0][j][2] != "neutral":
					#print("similar likes for " + i + " and " + q + ": " + str(likes_map[i][0][j]));
					simlikeq.append(i);
					found = True;
					break;
			if not found:
				for j in range(0, len(likes_map[i][1])):
					if likes_map[i][1][j] in likes_map[q][1] and likes_map[i][1][j][1] != "neutral":
						#print("similar likes for " + i + " and " + q + ": " + str(likes_map[i][1][j]));
						simlikeq.append(i);
						found = True;
						break;
		sim_likes[q] = simlikeq;

def onImport():
	fi = open("../data/extract_samples/pID_AEU");
	pid = fi.readlines();
	fi.close();
	for i in range(0, len(pid)):
		parts = pid[i].split("\t");
		s_set.add(parts[0]);
		s_map[parts[1].strip()] = parts[0];

if __name__ == "__main__":
	main();

if __name__ != "__main__":
	onImport();
