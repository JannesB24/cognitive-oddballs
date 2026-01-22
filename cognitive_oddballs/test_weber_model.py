
from models.weber_model import Weber_model
from environments.change_point_oddball_environment import generate_oddball_environment
from environments.random_walk_oddball_environment import generate_random_walk_environment
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

def compare_trajectories(m_1: Weber_model, m_2: Weber_model, node_idx: int, col_to_compare: str) -> pd.DataFrame:
    """Compares the given aspect of trajectories of the given Node in two models.
    Both Models must have the node to be compared
    
    Input:
    - m_1: First Model to be compared
    - m_2: Second Model to be compared
    - node_idx: Index of the node to be compared
    - col_to_compare: Possible values: "expected_mean","expected_precision","mean","precision","surprise" 
    
    Output:
    - DataFrame containing the values in question and the differences"""

    accepted_columns = ["expected_mean","expected_precision","mean","precision","surprise"]

    m1_df = m_1.to_pandas()
    m2_df = m_2.to_pandas()

    if len(m1_df.columns) < ((node_idx+1)*6 +3) or len(m2_df.columns) < ((node_idx+1)*6 +3):
        raise ValueError("Node with given Index must be present in both Models.")
    if col_to_compare not in accepted_columns:
        raise ValueError("col_to_compare must be one of the following: 'expected_mean','expected_precision','mean','precision','surprise'")
    
    col_name = str("x_"+str(node_idx)+"_"+col_to_compare)
    col_name_m1 = str("Model 1 Node " + str(node_idx) + " precision")
    col_name_m2 = str("Model 2 Node " + str(node_idx) + " precision")
    return pd.DataFrame({col_name_m1: m1_df[col_name], col_name_m2: m2_df[col_name], "Difference": m1_df[col_name]-m2_df[col_name]})


def compare_surprise(models: list):
    """Compares the over all surprise of node 0 of two given Models
    
    Input:
    - models: A list of models to be compared
    
    Output:
    - DataFrame containing the total surprises of node 0 for the models, the max surprise for each and which one has the lowest total surprise"""


    comparison = pd.DataFrame({"Model": range(len(models)), "Total_Surprise":range(len(models)), "Max_Surprise": range(len(models)), "Has_lowest_surprise":range(len(models))})
    for i in range(len(models)):
        current_model_df = models[i].to_pandas()
                      
        comparison.loc[i, "Model"] = ("Model "+str(i+1))
        comparison.loc[i, "Total_Surprise"] = sum(current_model_df["x_0_surprise"])
        comparison.loc[i, "Max_Surprise"] = max(current_model_df["x_0_surprise"])
        
        if i == 0:
            lowest_surprise = sum(current_model_df["x_0_surprise"])
        elif sum(current_model_df["x_0_surprise"]) < lowest_surprise:
            lowest_surprise = sum(current_model_df["x_0_surprise"])

    comparison["Has_lowest_surprise"] = comparison["Total_Surprise"] <= lowest_surprise

    return comparison
                 
            


## generating data from both environments
oddball_data = generate_oddball_environment(n_trials=1000, oddball_hazard_rate=0.15, sigma=20, change_point_hazard_rate=0.1, seed=42)
random_walk_data = generate_random_walk_environment(n_trials=1000, oddball_hazard_rate=0.15, sigma=20, seed=42)

### creating Model instances to test the current attempts
## with Node 4

# test_rw_low_p = Weber_model(random_walk_data,True)
# test_od_low_p = Weber_model(oddball_data,True)
# test_rw_high_p = Weber_model(random_walk_data,True,1e1)
# test_od_high_p = Weber_model(oddball_data,True,1e1)

## without Node 4 somewhat confusingly named

test_rw_3node = Weber_model(random_walk_data, node4=False)
test_od_3node = Weber_model(oddball_data, node4=False)


# ### comparing the surprises of different models in oddball environment
# surprise_comparison = compare_surprise([test_od_high_p, test_od_3node])
# print(surprise_comparison)
#
# ## RESULT
# ## Model with node 4 at high precision has lower surprise than Model without node 4. Highest Surprise is the same for both tho


### comparing hte surprise of Models without Node 4 across environments
#
surprise_comparison = compare_surprise([test_rw_3node, test_od_3node])
print(surprise_comparison)
#
### RESULT
## Model performs slightly better in Random walk environment. (Total surprise: 6118 vs. 6223)
## Even though the max surprise is a higher value in the random walk environment (Max surprise: 29 vs. 25)


# ### comparing the precision trajectories of node 3
#
#rw_precision_comp = compare_trajectories(test_rw_low_p,test_rw_high_p,3,"precision")
# od_precision_comp = compare_precisions(test_od_low_p,test_od_high_p,3,"precision")
#
# rw_precision_comp.to_csv("rw_precision_comp.csv")
# od_precision_comp.to_csv("od_precision_comp.csv")
#
# ## RESULTS
# ## higher precision of node 4 leads to higher precision of node 3 in the oddball environment
# ## opposite is true in the random walk environment


## extracting the node trajectories as data frames
#
# hp_df = test_rw_high_p.to_pandas()
# hp_df = test_od_high_p.to_pandas()
# lp_df = test_rw_low_p.to_pandas()
# lp_df = test_od_low_p.to_pandas()
#
# test_rw_3node_df = test_rw_3node.to_pandas()
# test_od_3node_df = test_od_3node.to_pandas()


### plotting the trajectories of the different model instances
#
# test_rw_low_p.plot_trajectories()
# test_od_low_p.plot_trajectories()
# test_rw_high_p.plot_trajectories()
# test_od_high_p.plot_trajectories()
#
# test_rw_3node.plot_trajectories()
# test_od_3node.plot_trajectories()


# ## checking the highest jump between observations (before the model cuts off)
#
# print("Highest jumps in low precision condition: ")
# print(" - Random Walk environment: " + str(test_rw_low_p.largest_jump()))
# print(" - Oddball environment: " + str(test_od_low_p.largest_jump())+ "\n")
#
# print("Highest jumps in high precision condition:")
# print(" - Random Walk environment: " + str(test_rw_high_p.largest_jump()))
# print(" - Oddball environment: " + str(test_od_high_p.largest_jump())+ "\n")
#
## without node 4
# print("Highest jumps without node 4: ")
# print(" - Random Walk environment: " + str(test_rw_3node.largest_jump()))
# print(" - Oddball environment: " + str(test_od_3node.largest_jump())+ "\n")
#
# # RESULT
# # high precision of node 4 leads the model to drop out at a smaller jump (/earlier) in the randomwalk environment (as compared to lower precision of node 4)
# # while high precision of node 4 leads the model to persevere after a overall larger jump in the oddball environment


## checking weird surprise spike in rw_3node
# print("Highest Surprise: ", test_rw_3node.max_total_surprise())
# test_rw_3node.plot_trajectories()
#
## RESULT
## High surprise at observation 504, because volatility was on a downward trajectory and suddenly jumped up when observation jumped from 269 to 6
## Surprise was less at the higher jump from 487 to 38 at observation 821, as the overall volatility was higher at that point anyway
#
## -> Maybe node 4 should be a value parent instead


### comparing node 0 mean trajectories and jumps
# comparing_mean = pd.DataFrame({"With Node 4 ":lp_df["x_0_mean"], "Without Node 4": test_rw_3node_df["x_0_mean"], "Is_Same": (lp_df["x_0_mean"]==test_rw_3node_df["x_0_mean"]), "Jump":range(len(lp_df))})
#
# for i in range(1,len(lp_df)):
#     comparing_mean.loc[i, "Jump"]= abs(comparing_mean.loc[i, "With Node 4 "]-comparing_mean.loc[i-1, "With Node 4 "])
#
# print(max(comparing_mean["Jump"]))
# comparing_mean.to_csv("Comparing_means.csv")



### saving stuff into cvs
##
# test_rw_low_p.to_pandas().to_csv("testing_rw_l.csv")
# test_od_high_p.to_pandas().to_csv("testing_od_h.csv")
# test_rw_3node_df.to_csv("testing_rw_without_n4.csv")





### OUTDATED basically creating a grid search to find a set of different node precisions that hold out the longest (by checking the number of NaN entries)
##    -> too inefficient
#
# best_precisions = [0,0,0,0,0]
# lowest_nan = 1000
# best_model = Weber_model(1,1,1,1,1)
# for i in range(11):
#     for j in range(11):
#         for k in range(11):
#             for l in range(11):
#                 for m in range(11):
#                     random_walk_model = Weber_model(i,j,k,l,m)
#                     random_walk_model.fit_to_random_walk_oddball_environment(random_walk_data)
#                     number_nan = np.count_nonzero(np.isnan(random_walk_model.to_pandas()))
#                     if (i == 1) and (j==1) and (k==1) and (l==1) and (m==1):
#                         best_precision = [1,1,1,1,1]
#                         lowest_nan = number_nan
#                         best_model = random_walk_model
#                     elif number_nan <= lowest_nan:
#                         best_precision = [i,j,k,l,m]
#                         lowest_nan = number_nan
#                         best_model = random_walk_model
#
# print( "Best precision: " + str(best_precision) + " With " + str(lowest_nan) + " NaN entries.")
# best_model.plot_trajectories()



### OUTDATED another grid search to find a model that does not fall out
##    -> none found
#
# test_model = Weber_model(1,1,1,1,1)
# valid_precisions = []
# for i in range(6):
#     for k in range(6):
#         for l in range(6):
#             for m in range(6):
#                 for n in range(6):
#                     test_model = Weber_model(n,m,l,k,i)
#                     test_model.fit_to_random_walk_oddball_environment(random_walk_data)
#                     number_nan = np.count_nonzero(np.isnan(test_model.to_pandas()))
#                     if number_nan == 0:
#                         valid_precisions.append((n,m,l,k,i))
# 
# print(valid_precisions)


### OUTDATED trying to find a tonic volatility of node 4 that helps performance
#
# test_model = Weber_model(3,1)
# best_model = Weber_model(3,1)
# valid_tv = []
# best_tv = 0
# lowest_nan = 0
#
# for i in range(11):
#     test_model = Weber_model(3,i)
#     test_model.fit_to_random_walk_oddball_environment(random_walk_data)
#     number_nan = np.count_nonzero(np.isnan(test_model.to_pandas()))
#    
#     if i == 1:
#         lowest_nan = number_nan
#         best_tv = 1
#         best_model = test_model
#
#     if number_nan == 0:
#         valid_tv.append(i)
#    
#     if number_nan < lowest_nan:
#         best_model = test_model
#         best_tv = i
#
# print(valid_tv)
# print("Closest Match with Tonic volatility: "+ str(best_tv))
