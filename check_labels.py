

import NLU

def main():
	t2()

def t2():
	f = open('crf-input-data')
	clines = f.readlines()
	f.close()
	u2 = list()
	utt = list()
	t2 = list()
	tutt = list()
	for cl in clines:
		parts = cl.strip()
		if parts == '':
			if utt != []:
				u2.append(utt)
				t2.append(tutt)
				utt = list()
				tutt = list()
		else:
			parts = parts.split()
			utt.append(parts[0])
			tutt.append(parts[2])
	if utt != []:
		u2.append(utt)
		t2.append(tutt)
		utt = list()
		tutt = list()

	lines = NLU.getALines()
	utterances = NLU.getUtterances(lines)
	for u in range(0, len(utterances)):
		slots = NLU.getSlots(utterances[u])
		sclist = list()
		for slot in slots[0]:
			sclist.append([slot[1], slot[2]])
		entlist = NLU.getEntities(u2[u], t2[u])[0]
		l1 = list()
		l2 = sclist
		for ent in entlist:
			l1.append([ent[1], ent[2]])
		if l1 != l2:
			print(str(l1) + '_' + str(l2))

def t1():
	f = open('EECS_annotated_samples_anonymized')
	lines = f.readlines()
	f.close()

	inst = 'n'
	cid = ''
	cname = ''

	broken = 0
	for i in lines:
		if i.startswith('<class'):
			inst = 'c'
		elif i.startswith('<instructor'):
			inst = 'i'
		if inst == 'c' and i.startswith('id='):
			cid = i.strip()
		elif inst == 'c' and i.startswith('name='):
			cname = i.strip()
		if i.endswith('>\n'):
			if inst == 'c':
				if cid != '' and cname != '':
					broken += 1
					print(cid + ':' + cname)
			inst = 'n'
			cid = ''
			cname = ''
	print(broken)


if __name__ == '__main__':
	main()
