from subprocess import *
import os.path
import numpy

def main():
	model = getModel("This is a test sentence.");
	test = lookup("This", model);
	print(test);

def lookup(word, model):
	nodes = model[0];
	rnodes = model[1];
	edges = model[2];
	sentipair = list();
	if word in nodes:
		sentipair = [nodes[word][0]];
		parent = edges[nodes[word][1]];
		if parent in rnodes:
			sentipair.extend([rnodes[parent][0]]);
		else:
			sentipair.extend([2]);
		if parent in edges:
			parent2 = edges[parent];
			#print(rnodes[parent2]);
			if parent2 in rnodes:
				sentipair.extend([rnodes[parent2][0]]);
			else:
				sentipair.extend([2]);
		else:
			sentipair.extend([2]);
	else:
		sentipair = [-1, 2, 2];
	#until i fix the word sentiment level
	#sentipair = sentipair[1:];
	return sentipair;

def getMappedModel(sentence):
	return internal_map[sentence];

def getModel(sentence):
	callout = "java -cp ../corenlp/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2.jar:../corenlp/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2-models.jar:../corenlp/stanford-corenlp-full-2015-04-20/ejml-0.23.jar:TreeMaker.jar edu.umich.cfwelch.TreeMaker";
	callout = callout.split(" ");
	callout.extend(["\"" + str(sentence) + "\""]);
	#print(callout);
	result = jarWrapper(callout);
	return modelFromLines(result);

def modelFromLines(result):
	nodes = dict();
	rnodes = dict();
	edges = dict();
	mode = 0;
	for i in range(0, len(result)):
		if "+++++++++++++++++++++++++++++++++" == result[i]:
			mode += 1;
		elif mode == 1:
			partz = result[i].split(":");
			rnodes[partz[2]] = [partz[1], partz[0]];
			nodes[partz[0]] = [partz[1], partz[2]];
		elif mode == 2:
			partz = result[i].split(":");
			edges[partz[0]] = partz[1];
	#print("nodes: " + str(nodes));
	#print("edges: " + str(edges));
	return [nodes, rnodes, edges];

def jarWrapper(args):
	process = Popen(args, stdout=PIPE, stderr=PIPE);
	ret = [];
	while process.poll() is None:
		line = process.stdout.readline();
		if line != '' and line.endswith('\n'):
			ret.append(line[:-1]);
	stdout, stderr = process.communicate();
	ret += stdout.split('\n');
	if stderr != '':
		ret += stderr.split('\n');
	ret.remove('');
	return ret;

def getListModel(sid):
	return internal_list[sid];

fo = open("treesout", "r");
lines = fo.readlines();
fo.close();
thesents = list();
curlist = list();
for q in range(0, len(lines)):
	if lines[q].strip() == "---------------------------------":
		if curlist:
			thesents.append(curlist);
			curlist = list();
	else:
		curlist.append(lines[q].strip());
if curlist:
	thesents.append(curlist);
print("sizeofthesents: " + str(len(thesents)));
print("el0: " + str(len(thesents[0])));
internal_map = dict();
internal_list = list();
for q in range(0, len(thesents)):
	mdl = modelFromLines(thesents[q][1:]);
	internal_map[thesents[q][0]] = mdl;
	internal_list.append(mdl);

if __name__ == "__main__":
	main();
