import dicts

f = open("EECS_annotated_samples_anonymized");
l = f.readlines();
f.close();
s = set();
claz = dicts.getEECSdict();
for i in l:
	t = i.strip();
	if t.startswith("link="):
		tid = -1;
		for cla in claz.keys():
			cname = t[5:];
			if cname.endswith(">"):
				cname = cname[:-1];
			if claz[cla] == cname:
				print(cla);
				tid = cla;
				break;
		if tid > 0:
			s.add(tid);
		else:
			iname = t[5:];
			if iname.endswith(">"):
				iname = iname[:-1];
			s.add(iname);
	elif t.startswith("id="):
		idx = t[3:];
		if idx.endswith(">"):
			idx = idx[:-1];
		s.add(idx);

print(s);
print(len(s));

