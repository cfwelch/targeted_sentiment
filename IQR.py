import sys

scores = open(sys.argv[1]);
lines = scores.readlines();
scores.close();

s = list();
for i in lines:
	s.append(float(i.strip()));

s.sort();
#print(s);
samples = len(s);

print("Q1 = " + str(s[len(s)/4-1]));
print("Q3 = " + str(s[len(s)*3/4-1]));
print("Samples = " + str(samples));
print("Mean = " + str(sum(s)/samples));
print("Median = " + str(s[len(s)/2-1]));
