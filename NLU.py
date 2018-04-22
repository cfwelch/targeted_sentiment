from stanford_corenlp_pywrapper import CoreNLP
import pycrfsuite
import crf_example
import entity_distance
import dicts
import pickle
import senti_lexis
import sklearn
import re

def main():
	lines = getALines();
	#only take 20% test sample
	fi = open("splits", "r");
	foldsin = fi.readlines();
	fi.close();
	# format folds inputs
	fsplits = open("splits");
	slines = fsplits.readlines();
	splits = list();
	for i in range(0, len(slines)):
		parts = slines[i].strip().split(":");
		train = list();
		test = list();
		for s in parts[0][1:-1].split(", "):
			train.append(int(s));
		for s in parts[1][1:-1].split(", "):
			test.append(int(s));
		splits.append((train, test));
	fsplits.close();
	Nfolds = len(splits);
	print("Number of folds: " + str(Nfolds));
	fdict = open("sentiment_dictionary", "r");
	cv = pickle.loads(fdict.read());
	fdict.close();
	instructor_comparisons = list();
	class_comparisons = list();
	foutput = open("NLU_scores", "w");
	for fold in range(0, Nfolds):
		#for evaluation
		Ctotal_correct = 0;
		Ctotal_guessed = 0;
		Ctotal_actual = 0;
		Itotal_correct = 0;
		Itotal_guessed = 0;
		Itotal_actual = 0;
		#get utterances
		in_utter = getUtterances(lines);
		take_utter = list();
		for i in range(0, len(in_utter)):
			if i in splits[fold][1]:
				take_utter.append(in_utter[i]);
		in_utter = take_utter;
		fclf = open("classifiers/sentiment_classifier" + str(fold), "r");
		clf = pickle.loads(fclf.read());
		fclf.close();
		#ftclf = open("taken_classifier", "r");
		#tclf = pickle.loads(ftclf.read());
		proc = CoreNLP("pos", corenlp_jars=["./dependencies/stanford-corenlp-full-2015-04-20/*"]);
		tagger = pycrfsuite.Tagger();
		tagger.open("taggers/advising_crf_tagger" + str(fold));
		#classify utterances
		for k in range(0, len(in_utter)):
			print("Current Utterance: " + in_utter[k][0]);
			#get slots from utterance
			slots = getSlots(in_utter[k]);
			print("SLOTS: " + str(slots));
			#constituency parse
			parsed = proc.parse_doc(in_utter[k][0]);
			#print(parsed);
			#print(str(list(parsed['sentences'][0]['tokens'])));
			print("\n\n\n");
			print("Number of parsed sentences: " + str(len(parsed['sentences'])));
			spos_tlist = list();
			for i in range(0, len(parsed['sentences'])):
				spos_tuples = zip(parsed['sentences'][i]['tokens'], parsed['sentences'][i]['pos']);
				spos_tlist.append(spos_tuples);
			X_test = [crf_example.sent2featuresWithSent(s, in_utter[k][0]) for s in spos_tlist];
			y_pred = [tagger.tag(xseq) for xseq in X_test];
			#print(parsed['sentences'][0]['tokens']);
			#print(y_pred[0]);
			classes = list();
			instructors = list();
			for i in range(0, len(parsed['sentences'])):
				etemp = getEntities(parsed['sentences'][i]['tokens'], y_pred[i]);
				for j in range(0, len(etemp[0])):
					classes.append([etemp[0][j][1], etemp[0][j][2]]);
				for j in range(0, len(etemp[1])):
					instructors.append(etemp[1][j]);
			print("Classes: " + str(classes));
			print("Instructors: " + str(instructors));
			csents = list();
			#ctakens = list();
			isents = list();
			for i in range(0, len(classes)):
				csents.append(getClassLabel(in_utter[k][0], classes[i], y_pred[0], parsed['sentences'][0]['tokens'], cv, clf));
				#ctakens.append(getClassLabel(in_utter[k][0], classes[i], y_pred[0], parsed['sentences'][0]['tokens'], cv, tclf));
			for i in range(0, len(instructors)):
				isents.append(getClassLabel(in_utter[k][0], [instructors[i]], y_pred[0], parsed['sentences'][0]['tokens'], cv, clf));
			#generate tuples for comparison of classes
			tlist = list();
			seval = list();
			#lists for printing recognized class comparison with actual
			cca = list();
			ccb = list();
			for q in range(0, len(slots[0])):
				Ctotal_actual += 1;
				cca.append((slots[0][q][1], slots[0][q][2], slots[0][q][3]));#slots[0][q][0] from this and next line
				tlist.append([slots[0][q][1], slots[0][q][2]]);
				seval.append(slots[0][q][3]);
			for i in range(0, len(classes)):
				Ctotal_guessed += 1;
				ccb.append((classes[i][0], classes[i][1], csents[i][0]));
				if classes[i] in tlist:
					if seval[tlist.index(classes[i])] == csents[i]:
						Ctotal_correct += 1;
			if (len(cca) > 0 or len(ccb) > 0) and cca != ccb:
				class_comparisons.append((in_utter[k][0], cca, ccb));
			#generate tuples for comparison of instructors
			tlist = list();
			seval = list();
			#lists for printing recognized instructor comparison with actual
			ica = list();
			icb = list();
			for q in range(0, len(slots[1])):
				Itotal_actual += 1;
				ica.append((slots[1][q][0], slots[1][q][1]));
				tlist.append(slots[1][q][0]);
				seval.append(slots[1][q][1]);
			for i in range(0, len(instructors)):
				Itotal_guessed += 1;
				icb.append((instructors[i], isents[i][0]));
				if instructors[i] in tlist:
					if seval[tlist.index(instructors[i])] == isents[i]:
						Itotal_correct += 1;
			if (len(ica) > 0 or len(icb) > 0) and ica != icb:
				instructor_comparisons.append((in_utter[k][0], ica, icb));
			#print output
			print("\n\nInput: " + in_utter[k][0]);
			print("Output: ");
			for i in range(0, len(classes)):
				print("Class: " + str(classes[i]) + " - " + str(csents[i]));# + " - " + str(ctakens[i]));
			for i in range(0, len(instructors)):
				print("Instructors: " + str(instructors[i]) + " - " + str(isents[i]));
		precision = (Ctotal_correct + Itotal_correct) * 1.0 / (Ctotal_guessed + Itotal_guessed);
		recall = (Ctotal_correct + Itotal_correct) * 1.0 / (Ctotal_actual + Itotal_actual);
		c_precision = Ctotal_correct * 1.0 / Ctotal_guessed;
		c_recall = Ctotal_correct * 1.0 / Ctotal_actual;
		i_precision = Itotal_correct * 1.0 / Itotal_guessed;
		i_recall = Itotal_correct * 1.0 / Itotal_actual;
		foutput.write(str([precision, recall, c_precision, c_recall, i_precision, i_recall]) + "\n");
		foutput.flush();
		break;
	foutput.flush();
	foutput.close();
	print("\nInstructor Comps: ");
	for i in range(0, len(instructor_comparisons)):
		print(instructor_comparisons[i]);
	#print("\nClass Comps: ");
	#for i in range(0, len(class_comparisons)):
	#	print(class_comparisons[i]);

def getUtterances(lines):
	full = list();
	temp = list();
	for i in range(0, len(lines)):
		if "\n" == lines[i]:
			full.append(temp);
			temp = list();
		else:
			temp.append(lines[i]);
	print("Utterances Parsed: " + str(len(full)));
	return full;

def getALines():
	fo = open("EECS_annotated_samples", "r");#_lies_links
	lines = fo.readlines();
	fo.close();
	return lines;

def getSlots(utterance):
	#False for NLU testing, True for sentiment testing
	USELINK = False;
	classes = list();
	instructors = list();
	mode = '';
	department = "";
	entity = "";
	eid = "";
	sentiment = "";
	link = "";
	for i in range(1, len(utterance)):
		if utterance[i].endswith("\n"):
			utterance[i] = utterance[i][:-1];
		#print("U:" + str(utterance[i]));
		if utterance[i].startswith("<class"):
			mode = 'c';
		elif utterance[i].startswith("<instructor"):
			mode = 'i';
		elif utterance[i].startswith("sentiment="):
			sentiment = utterance[i][10:];
			if sentiment.endswith(">"):
				sentiment = sentiment[:-1];
		elif utterance[i].startswith("id="):
			eid = utterance[i][3:];
			if eid.endswith(">"):
				eid = eid[:-1];
		elif utterance[i].startswith("department="):
			department = utterance[i][11:];
			if department.endswith(">"):
				department = department[:-1];
			department = department.upper();
		elif utterance[i].startswith("name="):
			entity = utterance[i][5:];
			if entity.endswith(">"):
				entity = entity[:-1];
		elif utterance[i].startswith("link="):
			link = utterance[i][5:];
			if link.endswith(">"):
				link = link[:-1];
		if utterance[i].endswith(">"):
			#print("LINK: " + str(link));
			if sentiment == "":
				sentiment = "neutral";
			if mode == 'c':
				if link != "" and USELINK:
					classes.append((department, eid, link, sentiment));
				else:
					classes.append((department, eid, entity, sentiment));
			else:
				if link != "" and USELINK:
					instructors.append((link, sentiment));
				else:
					instructors.append((entity, sentiment));
			department = "";
			entity = "";
			link = "";
			sentiment = "";
			eid = "";
	return (classes, instructors);

def getClassLabel(utterance, entity, BIO, tokens, cv, clf):
	regex = re.compile(r"[^a-zA-Z0-9_\~ ]+");

	# add token boundaries to the sentence
	tokenSent = utterance
	# use the ID, not the department...
	for tag in range(0, len(entity)):
		if "" != entity[tag]:
			print("eT:" + str(entity[tag]));
			tokenSent = tokenSent.replace(entity[tag], " ~~t~~ " + entity[tag]);
	#print(tokenSent);
	parts = regex.sub("", tokenSent).split(" ");

	# remove empty parts from the sentence
	while "" in parts:
		parts.remove("");

	# locate window feature indicies
	windowFeatures = [];
	done = False;
	while not done:
		for part in range(0, len(parts)):
			if "~~t~~" == parts[part]:
				windowFeatures += [part];
				parts.remove(parts[part]);
				#print("parts?: " + str(parts));
				break;
			if part == len(parts) - 1:
				done = True;
	print("window features: " + str(windowFeatures));

	for i in range(0, len(tokens)):
		tokens[i] = regex.sub("", tokens[i]);
	row = [];
	featureMapG = [[0]*300]*4;
	featureMap = {};
	Nflag = 0;
	for part in range(0, len(tokens)):
		thepart = tokens[part].lower();
		if thepart in cv:
			theid = cv.index(thepart);
			mindist = 999;
			for wf in range(0, len(windowFeatures)):
				distance = abs(windowFeatures[wf] - part);
				if distance < mindist:
					mindist = distance;
			mindist += 1;
			sentiz = senti_lexis.lexCounts(thepart);
			#for g_vi in range(0, len(g_vec)):
			#	featureMapG[0][g_vi] += g_vec[g_vi];# - mindist/10.0;
			#	featureMapG[1][g_vi] += g_vec[g_vi];# - mindist/10.0;
			#	featureMapG[2][g_vi] += g_vec[g_vi];# - mindist/10.0;
			#	featureMapG[3][g_vi] += g_vec[g_vi];# - mindist/10.0;
			if theid in featureMap:
				# 2.0 - mindist / 7.0 worked well for the first distance measure...
				# featureMap[theid] += 1.0 / mindist;
				featureMap[theid][0] += 2.0 - mindist / 7.0;
				featureMap[theid][1] += (2.0 - mindist / 7.0) * sentiz[0];
				featureMap[theid][2] += (2.0 - mindist / 7.0) * sentiz[1];
				featureMap[theid][3] += (2.0 - mindist / 7.0) * sentiz[2];
				if Nflag > 0:
					featureMap[theid][4] = 1.0;
			else:
				# featureMap[theid] = 1.0 / mindist;
				# count, positive, negative, neutral, negate
				featureMap[theid] = [0, 0, 0, 0, 0];
				featureMap[theid][0] = 2.0 - mindist / 7.0;
				featureMap[theid][1] = (2.0 - mindist / 7.0) * sentiz[0];
				featureMap[theid][2] = (2.0 - mindist / 7.0) * sentiz[1];
				featureMap[theid][3] = (2.0 - mindist / 7.0) * sentiz[2];
				if Nflag > 0:
					featureMap[theid][4] = 1.0;
			if Nflag > 0:
				Nflag -= 1;
			if senti_lexis.lexNegate(thepart):
				Nflag = 2;
	for i in range(0, len(cv)):
		if i in featureMap:
			row.extend(featureMap[i]);
		else:
			row.extend([0, 0, 0, 0, 0]);
	# add on the glove features
	for a in range(0, len(featureMapG)):
		temp_vec = [];
		for a_a in range(0, len(featureMapG[a])):
			temp_vec.append(featureMapG[a][a_a]*1.0/len(parts));
		row.extend(temp_vec);
	# make prediction
	return clf.predict(row);

def getEntities(tokens, BIO):
	depts = ["AAPTIS", "AAS", "ACABS", "AERO", "AEROSP", "AMCULT", "ANATOMY", "ANTHRARC", "ANTHRBIO", "ANTHRCUL", "AOSS", 
	"APPPHYS", "ARABAM", "ARCH", "ARMENIAN", "ARTDES", "ASIAN", "ASIANLAN", "ASIANPAM", "ASTRO", "AUTO", "BA", "BCS", "BE", 
	"BIOINF", "BIOLCHEM", "BIOLOGY", "BIOMATLS", "BIOMEDE", "BIOPHYS", "BIOSTAT", "BUDDHST", "CCS", "CDB", "CEE", "CHE", "CHEM", 
	"CHEMBIO", "CJS", "CLARCH", "CLCIV", "CLLING", "CMBIOL", "CMPLXSYS", "COGSCI", "COMM", "COMP", "COMPLIT", "CSP", "CZECH", 
	"DANCE", "DESCI", "DUTCH", "EARTH", "ECON", "EDCURINS", "EDUC", "EEB", "EECS", "EHS", "ELI", "ENGLISH", "ENGR", "ENS", 
	"ENSCEN", "ENVIRON", "ES", "ESENG", "EURO", "FRENCH", "GEOG", "GERMAN", "GREEK", "GTBOOKS", "HF", "HISTART", "HISTORY", 
	"HJCS", "HMP", "HONORS", "HS", "HUMGEN", "IMMUNO", "INSTHUM", "INTLSTD", "INTMED", "INTPERF", "IOE", "ITALIAN", "JAZZ", 
	"JUDAIC", "KINESLGY", "LACS", "LATIN", "LATINOAM", "LHSP", "LING", "MACROMOL", "MATH", "MATSCIE", "MCDB", "MECHENG", "MEDCHEM", 
	"MEMS", "MENAS", "MFG", "MICRBIOL", "MILSCI", "MKT", "MODGREEK", "MOVESCI", "MUSEUMS", "MUSICOL", "MUSMETH", "MUSPERF", 
	"MUSTHTRE", "NATIVEAM", "NAVARCH", "NAVSCI", "NERS", "NEUROSCI", "NRE", "NURS", "ORGSTUDY", "PAT", "PATH", "PHARMSCI", "PHIL", 
	"PHRMACOL", "PHYSICS", "PHYSIOL", "PIBS", "PMR", "POLISH", "POLSCI", "PORTUG", "PPE", "PSYCH", "PUBPOL", "RACKHAM", "RCARTS", 
	"RCCORE", "RCHUMS", "RCIDIV", "RCLANG", "RCNSCI", "RCSSCI", "REEES", "RELIGION", "ROMLANG", "ROMLING", "RUSSIAN", "SAC", "SAS", 
	"SCAND", "SEAS", "SI", "SLAVIC", "SM", "SOC", "SPANISH", "STATS", "STDABRD", "SURVMETH", "SW", "TCHNCLCM", "THEORY", "THTREMUS", 
	"TO", "UARTS", "UC", "UKR", "UP", "WOMENSTD", "WRITING", "YIDDISH"];
	classes = list();
	instructors = list();
	department = "";
	entity = "";
	eid = "";
	course = False;
	for i in range(0, len(tokens)):
		if "A" == BIO[i]:
			if entity != "" and not course:
				instructors.append(entity);
				entity = "";
			course = True;
			if tokens[i].upper() in depts:
				department = tokens[i].upper();
			else:
				if eid != "":
					classes.append(tuple([department, eid, entity]));
					entity = "";
				eid = tokens[i];
		elif "B" == BIO[i]:
			if entity != "":
				if course:
					classes.append(tuple([department, eid, entity]));
				else:
					instructors.append(entity);
				eid = "";
			course = True;
			entity = tokens[i];
		elif "Y" == BIO[i]:
			if "" != entity or "" != eid:
				if course:
					classes.append(tuple([department, eid, entity]));
				else:
					instructors.append(entity);
				eid = "";
				entity = "";
			course = False;
			entity = tokens[i];
		elif "I" == BIO[i]:
			entity += " " + tokens[i];
		elif "Z" == BIO[i]:
			entity += " " + tokens[i];
	if "" != entity or "" != eid:
		if course:
			classes.append(tuple([department, eid, entity]));
		else:
			instructors.append(entity);
		entity = "";
		eid = "";
	#no duplicates
	clist_trim = list();
	for class_ in classes:
		if class_ not in clist_trim:
			clist_trim.append(class_);
	return [clist_trim, instructors];
"""if entity in depts:
	department = entity;
elif "" != entity:
	if is_number(entity):
		classes.append(tuple([department, entity]));
	else:
		# figure out if name is class or instructor name
		etypes = entity_distance.entityDistance(entity);
		if etypes[0][0] < etypes [1][0]:
			classes.append(tuple([department, entity]));
		else:
			instructors.append(entity);"""

def is_number(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

if __name__ == "__main__":
	main();
