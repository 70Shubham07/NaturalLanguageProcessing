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


#Appending data for each patient to the dataframe

for pid in range(len( df_train["PID"].unique() )):

    new_df = df_train[ df_train["PID"] == list(df_train["PID"].unique())[pid] ].dropna()    #Create a dataframe dataframe pid'th patient.
    list_of_diseases = list( new_df["Event"].unique() )
    
       
    for i in list_of_diseases:
        counted = list(new_df["Event"]).count(i)
    
        that_dict[ i ] = counted


    #Now, appending the series !

    #For that, we need to find the top 10 first.
        
    that_dict_sorted = sorted( that_dict , key = that_dict.get, reverse = True )   #Returns a list of the keys that are sorted in descending order.

    that_dict_sorted = that_dict_sorted[0:10]
    #print(that_dict_sorted)

    another_dict = {}                                        #The dictionary that will contain disease codes as values 
    for i in range(10):
        another_dict["Event"+str(i+1)] = that_dict_sorted[i]
        

    sol_df = sol_df.append( pd.Series( data = another_dict , name = ( list(df_train["PID"].unique() )[pid]  )  ) )  #Appending to the dataframe.



print(sol_df.head())



sol_df.to_csv("shubhamkwwr5.csv")
    
    
'''
a = {}
for i in range(10):
    a["Event"+str(i+1)] = i
print(a)
'''
        
     


    

    
