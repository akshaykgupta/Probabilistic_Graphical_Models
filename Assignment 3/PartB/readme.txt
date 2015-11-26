Final Input Format
See sample ner_input.txt and pos_input.txt for input format of NER and PoS models respectively.

Final Output Format
See sample ner_output.txt and pos_output.txt for input format of NER and PoS models respectively.

What to submit?
Submit your best code (best if trained on all training data and not just on a subset). 
The code should not need to train again. You should submit only the testing code, after the models have been trained. That is, you should not need to access the training data anymore. 

Submit your code is in a .zip file named in the format <EntryNo>.zip. Make sure that when we run “unzip yourfile.zip” the following files are produced in the present working directory:
compile.sh
run.sh
Writeup.pdf (and not writeup.pdf, Writeup.doc, etc)

Your code will be run as 
./run.sh inputfile.txt pos_outputfile.txt ner_outputfile.txt

Feel free to add any file you may need to support your run.sh

The MODEL_outputfile.txt should have the same number of lines as inputfile.txt. And it should have two additional characters per token line (space and labeling). Make sure your code passes format checker (format_checker.py) before final submission.
Here is how to use the format_checker.py:
It takes 3 arguments: unlabelled file, labelled file and model (ner/pos)
python format_checker.py MODEL_input.txt MODEL_output.txt pos 

We will give you a test file (called inputfile.txt). You have to generate two files called 'pos_outputfile.txt' and 'ner_outputfile.txt', containing pos/ner tag in the specified format. 

We will evaluate your model by compute Macro F-measure using Fscore_MODEL.py. Following are the commands to use the script:
python Fscore_MODEL.py MODEL_output.txt MODEL_gold.txt

Your code should run on GCL machines. 
