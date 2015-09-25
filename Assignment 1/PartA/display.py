# Updated: Sunday Aug 23, 5 pm.
from sys import argv
from os import system
f0 = open(argv[1],'r') # bayesian network file
f1 = open(argv[2],'r') # output file
count = 1 # counts number of queries in output file
treeString = "digraph mentions {\n" # string describing tree in DOT language
numNodes = int(f0.readline())
# add title
treeString += 'node[shape=circle,fixedsize=true,width=0.6]\n' # all nodes should be circles

# Read network and write into treeString
for line in f0:
	[source,dest] = line.split()
	if dest == '[]':
		continue
	dest = dest.replace("[","").replace("]","").split(',')
	for d in dest:
		treeString += '"' + source + '"->"' + d + '"\n'
f0.close()

colors = ['red','green','cyan']
labels = ['query','obs','dsep','req-prob','req-obs']

# Read each line of output file
for line in f1:
	colorTree = treeString # colorTree stores string for coloring the nodes in DOT language
	outputList = line.split()
	if len(outputList) != 5:
		print "Error !!! In query number "+str(count) + ",wrong output format, program exiting !!!"
		exit()
	else:
		f2 = open(str(count)+".dot",'w') # Create dot file for each query
		
		# Create 5 clusters for creating legends in figures, 5 clusters are joined vertically by invisible edges
		colorTree += '\
		subgraph cluster_0 {\
		style=invis\
		b1 [shape=circle,label="query",style="filled",color=white,fixedsize=true,width=0.3];\
		a1 [shape=circle,label="",style=filled,fillcolor=red,width=0.4];\
		a1->b1[constraint=false,style=invis];\
	}\
	subgraph cluster_1 {\
		style=invis\
		b2 [shape=circle,label="observed",style="filled",color=white,fixedsize=true,width=0.3];\
		a2 [shape=circle,label="",style=filled,fillcolor=green,width=0.4];\
		a2->b2[constraint=false,style=invis];\
	}\
	subgraph cluster_2 {\
		style=invis\
		b3 [shape=circle,label="d-sep",style="filled",color=white,fixedsize=true,width=0.3];\
		a3 [shape=circle,label="",style=filled,fillcolor=cyan,width=0.4];\
		a3->b3[constraint=false,style=invis];\
	}\
	subgraph cluster_3 {\
		style=invis\
		b4 [shape=circle,label="req-prob",style="filled",color=white,fixedsize=true,width=0.3];\
		a4 [shape=doublecircle,label="",width=0.4];\
		a4->b4[constraint=false,style=invis];\
	}\
	subgraph cluster_4 {\
		style=invis\
		b5 [shape=circle,label="req-obs",style="filled",color=white,fixedsize=true,width=0.3];\
		a5 [shape=circle,label="*",width=0.4];\
		a5->b5[constraint=false,style=invis];\
	}\
	a1->a2[style=invis];\
	a2->a3[style=invis];\
	a3->a4[style=invis];\
	a4->a5[style=invis];\
	'
		nodes_dic = {} # stores which nodes are already seen, helps detecting duplicate nodes, key : node number, value : index of output label
		for i in range(5):
			output = outputList[i].split(':')
			if len(output)!=2:
				print "Error !!! In query number "+str(count) + ", wrong output format in label "+outputList[i]+", program exiting !!!"
				exit()
			[label,nodes] = output
			if label!=labels[i]:
				print "Error !!! In query number "+str(count) + ", wrong label, should have been " + labels[i]+" instead of "+label+", program exiting !!!"
				exit()
			if nodes != '[]':
				nodes = nodes.replace('[','').replace(']','').split(',')
				for n in nodes:
					# ignore evidence nodes in d-sep nodes
					if i == 2:
						if n in nodes_dic:
							if nodes_dic[n] == 1:
								continue
					if i < 3:
						if n in nodes_dic:
							print "Error !!! In query number "+str(count)+", node numbered "+n+" present in both "+str(labels[nodes_dic[n]])+" and "+str(labels[i])+", program exiting !!!"
							exit()
						nodes_dic[n] = i
						colorTree += '"' + n + '"' + '[shape=circle, style=filled, fillcolor='+colors[i]+']\n'
					elif i == 3:
						colorTree += '"' + n + '"' + '[shape=doublecircle]\n'
					else:
						colorTree += '"' + n + '"' + '[label="'+str(n)+'\n*'+'"]\n'
						
	
	colorTree += 'label="For clarity, only unobserved d-sep nodes are shown above\n";'
	colorTree += '}'		
	f2.write(colorTree)
	f2.close()
	system("dot -q -Teps "+str(count)+".dot -o "+str(count)+".eps") # generate figure from dot file. -Teps tells format of figure. To generate pdf, give argument -Tpdf, -q suppresses warnings
	#system("rm "+str(count)+".dot") # removes dot files
	count += 1
print "Figures successfully generated..."
