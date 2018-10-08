

import NLU, dicts
import string, re

def main():
	c1,g1,a1 = instructorLevel()
	c2,g2,a2 = classLevel()
	print((c1+c2)*1.0/(g1+g2))
	print((c1+c2)*1.0/(a1+a2))

def classLevel():
	CCOR = 0
	CGUE = 0
	CACT = 0
	pattern = re.compile("[\W_]+")
	w = dicts.getEECSdict()
	ww = list()
	for key in w.keys():
		ww.append(w[key])
	sentences = NLU.getALines()
	utterances = NLU.getUtterances(sentences)
	for u in utterances:
		xmatches = list()
		tutt = u[0].strip().lower()
		slots = NLU.getSlots(u)[0]
		for q in tutt.split():
			qq = pattern.sub("", q)
			if is_number(qq):
				xmatches.append(qq)
		for q in ww:
			if q.lower() in tutt:
				xmatches.append(q.lower())
		slist = list()
		for slot in slots:
			slist.append(slot[1].lower())
		print(slist)
		print(xmatches)
		CACT += len(slots)
		CGUE += len(xmatches)
		for name in xmatches:
			if name in slist:
				CCOR += 1
	print(str(CCOR*1.0/CGUE))
	print(str(CCOR*1.0/CACT))
	print(CACT)
	return CCOR, CGUE, CACT

def instructorLevel():
	ICOR = 0
	IGUE = 0
	IACT = 0
	profs = dicts.getProfWords()
	pattern = re.compile("[\W_]+")
	print profs
	sentences = NLU.getALines()
	utterances = NLU.getUtterances(sentences)
	for u in utterances:
		names = list()
		cname = ""
		slots = NLU.getSlots(u)[1]
		tutt = u[0].strip().lower().split()
		print slots
		for tok in tutt:
			ttok = pattern.sub("", tok)
			if ttok in profs:
				if cname != "":
					cname += " "
				cname += ttok
			else:
				if cname != "":
					names.append(cname)
				cname = ""
		if cname != "":
			names.append(cname)
		print(names)
		slist = list()
		for slot in slots:
			slist.append(slot[0].lower())
		IACT += len(slots)
		IGUE += len(names)
		for name in names:
			if name in slist:
				ICOR += 1
	print(str(ICOR*1.0/IGUE))
	print(str(ICOR*1.0/IACT))
	print(IACT)
	return ICOR, IGUE, IACT

def xactinst():
	w = dicts.getEECSprofs()
	sentences = NLU.getALines()
	utterances = NLU.getUtterances(sentences)
	xmatches = list()
	for i in utterances:
		tutt = i[0].strip().lower()
		for q in w:
			if q.lower() in tutt:
				xmatches.append(q)
	bees = len(xmatches)
	eyes = 0
	for x in xmatches:
		ptz = x.split()
		eyes += len(ptz) - 1
	print(bees)
	print(eyes)

def xactclass():
	w = dicts.getEECSdict()
	ww = list()
	for key in w.keys():
		ww.append(w[key])
	sentences = NLU.getALines()
	utterances = NLU.getUtterances(sentences)
	xmatches = list()
	for i in utterances:
		tutt = i[0].strip().lower()
		for q in ww:
			if q.lower() in tutt:
				xmatches.append(q)
	bees = len(xmatches)
	eyes = 0
	for x in xmatches:
		ptz = x.split()
		eyes += len(ptz) - 1
	print(bees)
	print(eyes)

def ruleBased():
	profs = dicts.getProfWords()
	eecs = dicts.getEECSWords()
	cbioyz = open('crf-input-data')
	l = cbioyz.readlines()
	cbioyz.close()
	wf, wl = getFL()
	guessy = 0
	guessz = 0
	guessb = 0
	guessi = 0
	guessa = 0
	cory = 0
	corz = 0
	corb = 0
	cori = 0
	cora = 0
	acty = 0
	actz = 0
	actb = 0
	acti = 0
	acta = 0
	PO = "O"
	for j in range(0, len(l)):
		parts = l[j].strip().split("\t")
		if len(parts) > 1:
			#count actuals
			if parts[2] == "Y":
				acty += 1
			elif parts[2] == "Z":
				actz += 1
			elif parts[2] == "A":
				acta += 1
			elif parts[2] == "B":
				actb += 1
			elif parts[2] == "I":
				acti += 1
			#check profs
			if parts[0].lower() in profs:
				if PO != "Y" and PO != "Z":
					guessy += 1
					if parts[2] == "Y":
						cory += 1
				elif PO == "Y" or PO == "Z":
					guessz += 1
					if parts[2] == "Z":
						corz += 1
			#check dept + number
			elif parts[0].lower() == "eecs" or is_number(parts[0]):
				guessa += 1
				if parts[2] == "A":
					cora += 1
			#check classes
			elif parts[0].lower() in wf and PO != "B" and PO != "I":
				#pnext = l[j+1].strip().split("\t")
				#if len(pnext) > 1 and pnext[0].lower() in wl:
				guessb += 1
				if parts[2] == "B":
					corb += 1
			elif parts[0].lower() in wl and (PO == "B" or PO == "I"):
				guessi += 1
				if parts[2] == "I":
					cori += 1
			PO = parts[2]
		else:
			PO = "O"
	py = 1.0*cory/guessy
	ry = 1.0*cory/acty
	pz = 1.0*corz/guessz
	rz = 1.0*corz/actz
	pa = 1.0*cora/guessa
	ra = 1.0*cora/acta
	pb = 1.0*corb/guessb
	rb = 1.0*corb/actb
	pi = 1.0*cori/guessi
	ri = 1.0*cori/acti
	#fscores
	fy = 2*py*ry/(py+ry)
	fz = 2*pz*rz/(pz+rz)
	fa = 2*pa*ra/(pa+ra)
	fb = 2*pb*rb/(pb+rb)
	fi = 2*pi*ri/(pi+ri)
	print(actb)
	print(acti)
	print(acty)
	print(actz)
	print(fy)
	print(fz)
	print(fa)
	print(fb)
	print(fi)

def getFL():
	w = dicts.getEECSdict()
	wf = list()
	wl = dict()
	for j in w.keys():
		ww = w[j].split()
		wf.append(ww[0].lower())
		if len(ww) > 1:
			wl[ww[0].lower()] = list()
			for i in range(1, len(ww)):
				wl[ww[0].lower()].append(ww[i].lower())
	return wf, wl

def is_number(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

if __name__ == "__main__":
	main()
