import os
import re
import pyparsing

def main():
	pp = parse("I have not yet taken but am looking forward to EECS 280. It seems like a great class.");
	print(treeDistance("class", "280", pp));

def parse(sentence):
	#sentence = "sometimes i see a bee shake. it is pretty cool to see.";
	#sentence = "I have not yet taken but am looking forward to EECS 280. It\"s a great class.";
	sentence = sentence.replace("\"", "\\\"");

	command = "echo \"" + sentence + "\" > /home/cfwelch/Documents/stanford-parser-full-2015-04-20/stanfordtemp.txt";
	#print("The command to be executed: " + command);
	os.popen(command);
	parser_out = os.popen("/home/cfwelch/Documents/stanford-parser-full-2015-04-20/lexparser.sh /home/cfwelch/Documents/stanford-parser-full-2015-04-20/stanfordtemp.txt").readlines();

	lineout = "".join([i for i in parser_out]);
	#print("LINEOUT: " + lineout);
	return lineout;

def treeDistance(word1, word2, parseTree):
	# initialize and use pyparse to generate nested list which is basically a tree
	depth = 0;
	pat = re.compile(r'(\(ROOT\n(.|\s[^\n])+\n\n)');
	match = pat.findall(parseTree);
	parens = pyparsing.nestedExpr( '(', ')');

	# merge the trees in the Stanford parse by common ROOT node
	tree = None;
	for m in match:
		temptree = parens.parseString(m[0]);
		if tree == None:
			tree = temptree.asList();
		else:
			aptree = temptree.asList()[0][1];
			#print("TEMPTREE: " + str(aptree));
			tree[0][1].append(aptree);
	#print("TREE: " + str(tree));

	# find the dependency tree distance between words from the merged tree
	treecopy = list(tree);
	first = treecopy;
	first.insert(0, 0);
	queue = first;
	#print("\nQUEUE: " + str(queue));
	mutualDepth = 0;
	bestA = 0;
	bestB = 0;
	while len(queue) > 0:
		depth = queue.pop(0);
		element = queue.pop(0);
		#print("\nDEPTH: " + str(depth));
		#print("\nELEMENT: " + str(element));
		if type(element) is not str:
			val1 = treeContains(word1, [list(element)]);
			val2 = treeContains(word2, [list(element)]);
			#print("VALS: " + str(val1) + ":" + str(val2));
			if val1 > -1 and val2 > -1 and mutualDepth < depth:
				mutualDepth = depth;
				bestA = val1;
				bestB = val2;
			for i in range(1, len(element)):
				queue.append(int(depth) + 1);
				queue.append(element[i]);
	distance = bestA + bestB;
	#print("Distance between " + word1 + " and " + word2 + " is " + str(distance) + ".");
	return distance;

def treeContains(word, tree):
	found = -1;
	first = tree;
	first.insert(0, 0);
	#print("\nFIRST: " + str(first));
	queue = first;
	#print("\nQUEUE: " + str(queue));
	while len(queue) > 0:
		#print("\n----------------------------------------------");
		depth = queue.pop(0);
		element = queue.pop(0);
		#print("\nDEPTH: " + str(depth));
		#print("\nTYPE: " + str(type(element)));
		#print("\nELC: " + str(element) + ":" + word);
		if str(element) == word:
			found = depth;
			break;
		elif type(element) is not str:
			for i in range(1, len(element)):
				#print("\nEL: " + str(element[i]));
				queue.append(int(depth) + 1);
				queue.append(element[i]);
	return found;

if __name__ == '__main__':
	main();
