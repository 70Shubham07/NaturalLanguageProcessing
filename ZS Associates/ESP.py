import pandas as pd

import time

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

sol_df = pd.DataFrame([], index = [])


the_strongest_event = {}
freq = 5
length = 7
for pid in range(len( df_train["PID"].unique() ) ):

    start = time.time()
    
    
    

    

    
    
    current_pid = list(df_train['PID'].unique())[pid]

    the_strongest_event[current_pid] = []
    
    
    #Creating the list of all possible updated sequences
    new_df = df_train[ df_train["PID"] == list(df_train['PID'].unique())[pid]].dropna()

    

    

    list_diseases = list(new_df["Event"])
    list_of_diseases = list(  new_df["Event"].unique() )

    #I have to repeat everything from this point on, 10 times.
    for the_strongest_event_counter in range(10):

        #To save computational time, I am taking out 15/35 portion of the beginning of the sequence
        list_diseases = list_diseases[( len(list_diseases) - int( len(list_diseases)/3.5 ) ) :] 
        
        

        #the_strength is being taken again, because we have to calculate new strengths from 50 to 90 after addition of previous
        # strongest event to the current sequence
        the_strength = {}

        

        #All possible extensions will be done again.
        possible_extensions = []

    

    


    
    
        possible_extensions_counter = 0

        #By end of this for loop, I'll have strengths of each disease in the_strength dictionary
        for k in list_of_diseases:
            
            possible_extensions.append( list_diseases + [k]  )

            weight=0
            #By the end of this for loop, I'll have strength of k in weight variable

            
            permutations = zip( *( list_diseases[i:] for i in range(length)  )  )
            permutations = list(permutations)

            i_limit = len(possible_extensions[possible_extensions_counter]) - length + 1

            #By end of this for loop, I'll have the frequent lists and the
            #accumulated weights of all frequent subsequences that have the length of current length
            for perm in permutations:
                count=0

                for i in range(i_limit):
                    if( list(perm) == list_diseases[i:i+length]  ):
                        count+=1

                if(count>=freq):
                    #Patient_dict[current_pid][str(length)].append(list(k))
                    weight = weight+(count*length)

                    

            the_strength[k] = weight
        
            
            possible_extensions_counter+=1

        #Taking the event with the maximum strength.
        the_strongest_event[current_pid].append( max( the_strength ) )

        #Updatin the sequence of events of current patient by appending the predicted event.
        list_diseases = list_diseases + [ the_strongest_event[current_pid][the_strongest_event_counter] ]
        
        

    
    


        
    
    end = time.time()

    print(pid, end - start)
    

    #the_strongest_event is my answer.
    #Now, I have to repeat the entire process for same patient ID a total of 10 times, with this event added to list_of_diseases.

   
            
                    

    
    
    
#Now, constucting the dataframe. Then, wrtitng it to the csv file for submission.


for i in range( len(df_train["PID"].unique()) ):
    k=0
    another_dict = {}
    for k in range(10):

        another_dict["Event"+str(k+1)] = the_strongest_event[ list(df_train['PID'].unique())[i]  ][k]
        

    sol_df = sol_df.append( pd.Series( data = another_dict , name = ( list(df_train["PID"].unique() )[i]  )  ) )    





Event_10_col = sol_df["Event10"]
del sol_df["Event10"]
sol_df["Event10"] = Event_10_col


sol_df.index.name = "PID"

sol_df.to_csv("shubham.csv")

    
    
    

  
                


            
    
            

    
