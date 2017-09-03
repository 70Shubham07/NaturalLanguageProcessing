import pandas as pd
df_train = pd.read_csv("train.csv")

lengths = {}
for pid in range(len( df_train["PID"].unique() ) ):

    #start = time.time()
    #This list will have a length of 10.
    the_strongest_event = {}

    

    #the_strength = {}
    
    current_pid = list(df_train['PID'].unique())[pid]

    the_strongest_event[current_pid] = []
    #Patient_dict[current_pid] = {}
    
    #Creating the list of all possible updated sequences
    new_df = df_train[ df_train["PID"] == list(df_train['PID'].unique())[pid]].dropna()

    #possible_extensions = []

    

    list_diseases = list(new_df["Event"])
    #list_of_diseases = list(  new_df["Event"].unique() )
    lengths[current_pid] = len(list_diseases)

maxim = max(lengths)
minim = min(lengths)
print(lengths[maxim], lengths[minim])
