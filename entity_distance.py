from nltk.metrics import *
import dicts
import re

#from gensim.parsing.preprocessing import STOPWORDS
STOPWORDS = """
a about above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be
became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by call can
cannot cant co computer con could couldnt cry de describe
detail did didn do does doesn doing don done down due during
each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen
fify fill find fire first five for former formerly forty found four from front full further get give go
had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however hundred i ie
if in inc indeed interest into is it its itself keep last latter latterly least less ltd
just
kg km
made make many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely
neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off
often on once one only onto or other others otherwise our ours ourselves out over own part per
perhaps please put rather re
quite
rather really regarding
same say see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system take ten
than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two un under
until up unless upon us used using
various very very via
was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet you
your yours yourself yourselves
"""
STOPWORDS = frozenset(w for w in STOPWORDS.split() if w)

def main():
	# basic string distance
	string1 = "video game";
	string2 = "computer game design";
	print(edit_distance(string1, string2, transpositions=True));
	print(edit_distance(string1.lower(), string2.lower(), transpositions=True));
	print(wordDistance(string1, string2));
	print(entityDistance(string1));
	print(entityDistance("best"));
	print(entityDistance("professor"));

#Return N for stopword or not close to entity
#Return I for within 2 of instructor name
#Return C for within 2 of class name
def nearestEntity(inword):
	# don't use stopwords
	if inword in STOPWORDS:
		return "N";
	profs = dicts.getProfWords();
	for i in range(0, len(profs)):
		if edit_distance(inword, profs[i], transpositions=False) < 3:
			return "I";
	# class words add noise?
	classes = dicts.getEECSWords();
	removes = list();
	for i in range(0, len(classes)):
		if classes[i] in STOPWORDS:
			removes.append(classes[i]);
	for i in removes:
		#print("removing stopword " + i);
		classes.remove(i);
	for i in range(0, len(classes)):
		if edit_distance(inword, classes[i], transpositions=False) <3:
			return "C";
	return "N";

def entityDistance(string1):
	# prof or class
	eecs_profs = dicts.getEECSprofs();
	eecs_classes = dicts.getEECSdict();
	prof_distance = 999;
	prof_distance2 = 999;
	prof_nearest = "";
	class_distance = 999;
	class_distance2 = 999;
	class_nearest = "";
	profs_near = list();
	classes_near = list();
	# this finds nearest prof
	for i in range(0, len(eecs_profs)):
		pdist = minWordDistance(string1, eecs_profs[i]);
		if pdist < prof_distance:
			prof_distance = pdist;
	for i in range(0, len(eecs_profs)):
		pdist = minWordDistance(string1, eecs_profs[i]);
		if pdist == prof_distance:
			#print(eecs_profs[i]);
			profs_near.append(eecs_profs[i]);
	prof_distance2 = prof_distance
	prof_distance = 999;
	for i in range(0, len(profs_near)):
		pdist = abs(numParts(profs_near[i]) - numParts(string1));
		if pdist < prof_distance:
			prof_distance = pdist;
			prof_nearest = profs_near[i];
	# this finds nearest class
	eecs_cnames = eecs_classes.values();
	for i in range(0, len(eecs_cnames)):
		cdist = minWordDistance(string1, eecs_cnames[i]);
		if cdist < class_distance:
			class_distance = cdist;
	for i in range(0, len(eecs_cnames)):
		cdist = minWordDistance(string1, eecs_cnames[i]);
		if cdist == class_distance:
			classes_near.append(eecs_cnames[i]);
	class_distance2 = class_distance;
	class_distance = 999;
	for i in range(0, len(classes_near)):
		cdist = abs(numParts(classes_near[i]) - numParts(string1));
		if cdist < class_distance:
			class_distance = cdist;
			class_nearest = classes_near[i];
	# print nearest entities
	#print("CLASS NEAR: " + class_nearest + "(" + str(class_distance2) + ":" + str(class_distance) + ")");
	#print("PROF NEAR: " + prof_nearest + "(" + str(prof_distance2) + ":" + str(prof_distance) + ")");
	return [[class_distance2 + class_distance, class_nearest], [prof_distance2 + prof_distance, prof_nearest]];

def numParts(string1):
	pattern = re.compile("[^a-zA-Z0-9_ \t\n\r\f\v]+");
	string1 = pattern.sub("", string1);
	sparts1 = string1.lower().split(" ");
	return len(sparts1);

def minWordDistance(string1, string2):
	pattern = re.compile("[^a-zA-Z0-9_ \t\n\r\f\v]+");
	string1 = pattern.sub("", string1);
	string2 = pattern.sub("", string2);
	sparts1 = string1.lower().split(" ");
	sparts2 = string2.lower().split(" ");
	mind = 999;
	for i in range(0, len(sparts1)):
		for j in range(0, len(sparts2)):
			tmind = edit_distance(sparts1[i], sparts2[j], transpositions=False);
			if tmind < mind:
				mind = tmind;
	return mind;

def wordDistance(string1, string2):
	pattern = re.compile("[^a-zA-Z0-9_ \t\n\r\f\v]+");
	string1 = pattern.sub("", string1);
	string2 = pattern.sub("", string2);
	sparts1 = string1.lower().split(" ");
	sparts2 = string2.lower().split(" ");
	tscore = 0;
	for i in range(0, len(sparts1)):
		for j in range(0, len(sparts2)):
			tscore += edit_distance(sparts1[i], sparts2[j], transpositions=False);
	tscore = tscore / (len(sparts1) + len(sparts2));
	return tscore;

if __name__ == "__main__":
	main();
