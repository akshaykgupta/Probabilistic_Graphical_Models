import sys
import os,re
import pdb

# takes 3 arguments: unlabelled file, labelled file and model (ner/pos)
#python format_checker_Assignement3.py pos_input.txt pos_output.txt pos // to verify output of pos model
#python format_checker_Assignement3.py ner_input.txt ner_output.txt ner // to verify output of ner model

wo_lbl = open(sys.argv[1],"r")
w_lbl = open(sys.argv[2],"r")
model= sys.argv[3]

wo_lbl_lines = wo_lbl.readlines()
w_lbl_lines = w_lbl.readlines()

if len(wo_lbl_lines)!=len(w_lbl_lines):
	print "damn!!! number of lines incorrect... (error!!!)"
	exit(1)

#person', `geo-loc', `facility', `company',`product', `other

for i in range(len(wo_lbl_lines)):
	wo_items = wo_lbl_lines[i].strip().split(" ")
	w_items = w_lbl_lines[i].strip().split(" ")

	if wo_items[0]=="" and w_items[0]=="":
		continue

	if wo_items[0]=="" and w_items[0]!="":
		print "no way!!! line %d in labelled file should have been blank... (error!!!)" %(i+1)
		exit(1)

	if len(wo_items)!=1:
		print "oh no!!! unlabelled file doesn't have exactly 1 token in line %d... (error!!!)" %(i+1)
		exit(1)

	if len(w_items)!=2:
		print "oh no!!! labelled file doesn't have exactly 2 tokens in line %d... (error!!!)" %(i+1)
		exit(1)

	if w_items[0]!=wo_items[0]:
		print "woops!!! token mismatch, between labelled and unlabelled files, on line %d... (error!!!)" %(i+1)
		exit(1)

	if model=="ner":
		w_items[1]=re.sub("^B-","",w_items[1]);w_items[1]=re.sub("^I-","",w_items[1])
		if w_items[1]!="person" and w_items[1]!="geo-loc" and w_items[1]!="facility" and w_items[1]!="company" and w_items[1]!="product" and w_items[1]!="other" and w_items[1]!="O":
			print "darn!!! label token is different from person, geo-loc, facility, company, product, other, O on line %d... (error!!!)" %(i+1)
			exit(1)
	elif model=="pos":
		if w_items[1]!="!" and w_items[1]!="#" and w_items[1]!="$" and w_items[1]!="&" and w_items[1]!="," and w_items[1]!="A" and w_items[1]!="@" and w_items[1]!="E" and w_items[1]!="D" and w_items[1]!="G" and w_items[1]!="M" and w_items[1]!="L" and w_items[1]!="O" and w_items[1]!="N" and w_items[1]!="P" and w_items[1]!="S" and w_items[1]!="R" and w_items[1]!="U" and w_items[1]!="T" and w_items[1]!="V" and w_items[1]!="Y" and w_items[1]!="X" and w_items[1]!="Z" and w_items[1]!="^" and w_items[1]!="~":
			print "darn!!! label token is different from !, #, $, &, ,, A, @, E, D, G, M, L, O, N, P, S, R, U, T, V, Y, X, Z, ^, ~ on line %d... (error!!!)" %(i+1)
			exit(1)


print "congratulations!! your format is spot on!!!"

wo_lbl.close()
w_lbl.close()
