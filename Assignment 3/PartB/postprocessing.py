import sys
f = open(sys.argv[1])
g = open(sys.argv[2])
h = open(sys.argv[3],'w')
if len(sys.argv) > 4:
	action = sys.argv[4]
else:
	action = ''
for line in f:
	label = g.readline().strip()
	line = line.strip().split()
	if len(line) >= 1 and action == '':
		h.write(line[0]+' '+label+'\n')
	elif action == '':
		h.write('\n')
	if len(line) >= 1 and action == 'train':
		for l in line[:-1]:
			h.write(l + ' ')
		h.write(label + ' ')
		h.write(line[-1] + '\n')
	elif action == 'train':
		h.write('\n')
	if len(line) >= 1 and action == 'test':
		for l in line[:-1]:
			h.write(l + ' ')
		h.write(line[-1] + ' ' + label + '\n')
	elif action == 'test':
		h.write('\n')
f.close()
g.close()
h.close()