import sys

def main():
	# initalize program
	fo = open("EECS_annotated_samples", "r");
	lines = fo.readlines();
	mode = False;
	typeMode = 'o';
	instances = 0;
	classes = 0;
	instructors = 0;
	taken = 0;
	iname = 0;
	cname = 0;
	#sentiment values
	positive = 0;
	negative = 0;
	neutral = 0;
	isentiment = 0;
	csentiment = 0;
	ineutral = 0;
	cneutral = 0;
	passed = 0;
	instructor = 0;
	cid = 0;
	department = 0;
	performance = 0;
	difficulty = 0;
	want_to_take = 0;
	semester = 0;
	slots = 0;
	index = 0;
	links = 0;
	ilinks = 0;
	lname = "";
	haslink = False;
	# taken values
	takenTrue = 0;
	takenFalse = 0;
	takenUnknown = 0;
	takenLast = "";

	# parse file
	for i in range(len(lines)):
		data = lines[i].strip();
		if data.startswith("<"):
			mode = True;
			instances += 1;
		if "" == data:
			index += 1;
			if index == 119:
				# 20 % train test split - must be updated when more utterances are added
				print("Test Split: " + str(instances));
		if mode:
			slots += 1;
			#print data;
			if data == "<class":
				classes += 1;
				typeMode = 'c';
			elif data == "<instructor":
				instructors += 1;
				typeMode = 'i';
			#elif "<" in data and data != "<class" and data != "<instructor":
			#	print data;
			if data.startswith("taken"):
				taken += 1;
				takenLast = data[6:];
				if takenLast.endswith(">"):
					takenLast = takenLast[:-1];
			elif data.startswith("name"):
				lname = data;
				if typeMode == 'c':
					cname += 1;
				elif typeMode == 'i':
					iname += 1;
			elif data.startswith("sentiment"):
				if typeMode == 'c':
					csentiment += 1;
					tsent = data[10:];
					if tsent.endswith(">"):
						tsent = tsent[:-1];
					if tsent == "neutral":
						cneutral += 1;
						neutral += 1;
					elif tsent == "positive":
						positive += 1;
					elif tsent == "negative":
						negative += 1;
				elif typeMode == 'i':
					isentiment += 1;
					tsent = data[10:];
					if tsent.endswith(">"):
						tsent = tsent[:-1];
					if tsent == "neutral":
						ineutral += 1;
						neutral += 1;
					elif tsent == "positive":
						positive += 1;
					elif tsent == "negative":
						negative += 1;
			elif data.startswith("passed"):
				passed += 1;
			elif data.startswith("instructor"):
				instructor += 1;
			elif data.startswith("id"):
				cid += 1;
			elif data.startswith("department"):
				department += 1;
			elif data.startswith("performance"):
				performance += 1;
			elif data.startswith("difficulty"):
				difficulty += 1;
			elif data.startswith("want-to-take"):
				want_to_take += 1;
			elif data.startswith("semester"):
				semester += 1;
			elif data.startswith("link"):
				links += 1;
				#if haslink == True:
				#	print(lname);
				haslink = True;
			#else:
			#	print data + "!";
		if data.endswith(">"):
			if not haslink and lname != "":
				print("Last name: " + lname);
			haslink = False;
			lname = "";
			if typeMode == 'c':
				if takenLast == "true":
					takenTrue += 1;
				elif takenLast == "false":
					takenFalse += 1;
				else:
					takenUnknown += 1;
				takenLast = "";
			mode = False;
	slots -= instances;

	# print stats
	print "Instances: " + str(instances);
	print "Classes: " + str(classes);
	print "Instructors: " + str(instructors);
	print "Utterances: " + str(index);
	print "";
	print "Slots: " + str(slots);
	print "Positive: " + str(positive);
	print "Negative: " + str(negative);
	print "Neutral: " + str(neutral);
	print "Taken: " + str(taken) + ":" + str(taken*1.0/slots);
	print "Taken Values: True(" + str(takenTrue) + ") False(" + str(takenFalse) + ") Unknown(" + str(takenUnknown) + ")";
	print "Instructor Name: \t" + str(iname) + ":" + str(iname*1.0/slots);
	print "Instructor Sentiment: \t" + str(isentiment) + ":" + str(isentiment*1.0/slots);
	print "Instructor Neutral: \t" + str(ineutral);
	print "Class Name: \t\t" + str(cname) + ":" + str(cname*1.0/slots);
	print "Class Sentiment: \t" + str(csentiment) + ":" + str(csentiment*1.0/slots);
	print "Class Neutral: \t\t" + str(cneutral);
	print "Passed: \t\t" + str(passed) + ":" + str(passed*1.0/slots);
	print "Instructor: \t\t" + str(instructor) + ":" + str(instructor*1.0/slots);
	print "ID: \t\t\t" + str(cid) + ":" + str(cid*1.0/slots);
	print "Department: \t\t" + str(department) + ":" + str(department*1.0/slots);
	print "Performance: \t\t" + str(performance) + ":" + str(performance*1.0/slots);
	print "Difficulty: \t\t" + str(difficulty) + ":" + str(difficulty*1.0/slots);
	print "Want to take: \t\t" + str(want_to_take) + ":" + str(want_to_take*1.0/slots);
	print "Semester: \t\t" + str(semester) + ":" + str(semester*1.0/slots);
	print "Entity Links: \t\t" + str(links) + " of " + str(cname + iname);

if __name__ == "__main__":
	main();
