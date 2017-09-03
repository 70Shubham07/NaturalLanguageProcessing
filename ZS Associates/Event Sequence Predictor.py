'''
I have have the data for 3000 patients!

The corresponding columns are events

'''



import pandas as pd
#import operator

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")



sol_df = pd.DataFrame(  [] ,  index = [])   # The dataframe to which we'll upload our answers.


that_dict = {} #Will be used to create the series that will be appended to the sol_df dataframe.

a = []
b = []
#Appending data for each patient to the dataframe

for pid in range(len( df_train["PID"].unique() )):

    new_df = df_train[ df_train["PID"] == list(df_train["PID"].unique())[pid] ].dropna()
    list_of_diseases = list( new_df["Event"].unique() )
    
    
    for i in list_of_diseases:
        counted = list(new_df["Event"]).count(i)
        that_dict[i] = counted
        
             
             
             
             
             
    
    
    
        
    that_dict_sorted = sorted( that_dict , key = that_dict.get, reverse = True )
    
    that_dict_sorted = that_dict_sorted[0:10]
    #print(that_dict_sorted)

    another_dict = {}  
    for i in range(10):
        another_dict["Event"+str(i+1)] = that_dict_sorted[i]
        

    sol_df = sol_df.append( pd.Series( data = another_dict , name = ( list(df_train["PID"].unique() )[pid]  )  ) )
    
    a.append(list_of_diseases)
    b.append(list(new_df["Event"])
Event_10_col = sol_df["Event10"]
del sol_df["Event10"]
sol_df["Event10"] = Event_10_col


#Naming the index
sol_df.index.name = "PID"


#print(sol_df.head())


#Writing data to a csv file!
sol_df.to_csv("shubhamkwwr5.csv")
    

        '''
     

for i in list_of_diseases:
             counted = list(new_df["Event"]).count(i)
             that_dict[i] = couted
    
counted = list(new_df["Event"]).count(i)
that_dict[i] = counted
    
