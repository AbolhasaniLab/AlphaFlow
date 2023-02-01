This software runs the AlphaFlow algorithm for either the sequence selection or volume and time optimization campaigns. The algorithm is pretrained on real-world collected data sets that are included
in the "Test Scripts" folder, and it modifies an active input file to provide suggestions for the next test conditions. The algorithm will then wait until the active reward folder is updated to reflect the 
suggested test conditions, and a new condition is then suggested. During regular system operation, the reward file is automatically updated by the platform control software; however, it may be updated
manually for testing purposes.

Instructions for running program:
1. Open the files "Reward Active.txt" and "Inputs Active.txt", and check that the values are initialized. 

The Reward file should contain only:
"Reward
0"

The Inputs file should contain only:
"Inputs
0,0,0"

2. Run the desired primary script (either "Run Volume Time Optimization.py" or "Run Sequence Selection" or "Run Sequence Selection Notebook")

3. After the script has added a new line to the active input file, the user may modify the active reward file as desired by inserting scalar values line-by-line and saving. 
Note that the volume and time optimizationscript will not continue until a reward value has been provided for every oscillation of the suggested input conditions.