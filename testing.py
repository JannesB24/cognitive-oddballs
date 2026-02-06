from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment
from cognitive_oddballs.models.weber_model import WeberModel as Weber_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    if len(m1_df.columns) < ((node_idx+1)*6 +4) or len(m2_df.columns) < ((node_idx+1)*6 +4):
        raise ValueError("Node with given Index must be present in both Models.")
    if col_to_compare not in accepted_columns:
        raise ValueError("col_to_compare must be one of the following: 'expected_mean','expected_precision','mean','precision','surprise'")
    
    col_name = str("x_"+str(node_idx)+"_"+col_to_compare)
    col_name_m1 = str("Model 1 Node " + str(node_idx) + " " + col_to_compare)
    col_name_m2 = str("Model 2 Node " + str(node_idx) + " " + col_to_compare)
    return pd.DataFrame({col_name_m1: m1_df[col_name], col_name_m2: m2_df[col_name], "Difference": m1_df[col_name]-m2_df[col_name]})


def compare_surprise(models: list):
    """Compares the over all surprise of node 0 of two given Models
    
    Input:
    - models: A list of models to be compared
    
    Output:
    - DataFrame containing the total surprises of node 0 for the models, the max surprise for each, whether it cuts of and which one has the lowest total surprise (excluding models which cut off)"""


    comparison = pd.DataFrame({"Model": [" "]*len(models), "Total_Surprise":[1.1]*len(models), "Max_Surprise": [1.1]*len(models), "Cuts_off":  [False]*len(models) , "Has_lowest_surprise":[False]*len(models)})
    for i in range(len(models)):
        current_model_df = models[i].to_pandas()
                      
        comparison.loc[i, "Model"] = ("Model "+str(i+1))
        comparison.loc[i, "Total_Surprise"] = sum(current_model_df["x_0_surprise"])
        comparison.loc[i, "Max_Surprise"] = max(current_model_df["x_0_surprise"])
        comparison.loc[i, "Cuts_off"] = models[i].drops_out()

    lowest_surprise = np.nanmax(comparison["Total_Surprise"])
    lowest_model = len(models)
    for k in range(len(models)):
        if comparison.loc[k,"Total_Surprise"] <= lowest_surprise and not comparison.loc[k, "Cuts_off"]:
            lowest_surprise = comparison.loc[k,"Total_Surprise"]
            lowest_model = k
    comparison.loc[lowest_model,"Has_lowest_surprise"] = True

    return comparison


# # ## generating data from both environments
change_point_data = generate_change_point_environment(n_trials=1000, oddball_hazard_rate=0.15, sigma=20, change_point_hazard_rate=0.1, seed=42)
random_walk_data = generate_random_walk_environment(n_trials=1000, oddball_hazard_rate=0.15, sigma=20, seed=42)


### creating Model instances to test the current attempts

## 5 nodes (different precisions)
test_rw_low_p = Weber_model(n_nodes=5,x_4_p=3).input_data(random_walk_data["x"].to_numpy())
test_cp_low_p = Weber_model(n_nodes=5,x_4_p=3).input_data(change_point_data["x"].to_numpy())
test_rw_high_p = Weber_model(n_nodes=5).input_data(random_walk_data["x"].to_numpy())
test_cp_high_p = Weber_model(n_nodes=5).input_data(change_point_data["x"].to_numpy())

## 5 nodes with "standard" as the update type
test_rw_low_p_s = Weber_model(n_nodes=5,x_4_p=3, update_type="standard").input_data(random_walk_data["x"].to_numpy())
test_cp_low_p_s = Weber_model(n_nodes=5,x_4_p=3, update_type="standard").input_data(change_point_data["x"].to_numpy())
test_rw_high_p_s = Weber_model(n_nodes=5, update_type="standard").input_data(random_walk_data["x"].to_numpy())
test_cp_high_p_s = Weber_model(n_nodes=5, update_type="standard").input_data(change_point_data["x"].to_numpy())

## 4 nodes
test_rw_4_nodes = Weber_model(n_nodes=4).input_data(random_walk_data["x"].to_numpy())
test_cp_4_nodes = Weber_model(n_nodes=4).input_data(change_point_data["x"].to_numpy())

## 4 nodes with "standard" as the update type
test_rw_4_nodes_s = Weber_model(n_nodes=4, update_type="standard").input_data(random_walk_data["x"].to_numpy())
test_cp_4_nodes_s = Weber_model(n_nodes=4, update_type="standard").input_data(change_point_data["x"].to_numpy())

## 3 nodes
test_rw_3_nodes = Weber_model(n_nodes=3).input_data(random_walk_data["x"].to_numpy())
test_cp_3_nodes = Weber_model(n_nodes=3).input_data(change_point_data["x"].to_numpy())


# ### comparing the surprises of the test models in the change point environment
# cp_surprise_comparison = compare_surprise([test_cp_high_p, test_cp_high_p_s, test_cp_low_p,test_cp_low_p_s, test_cp_4_nodes,test_cp_4_nodes_s, test_cp_3_nodes])
# cp_surprise_comparison["Model"] = ["test_cp_high_p","test_cp_high_p_s", "test_cp_low_p","test_cp_low_p_s", "test_cp_4_nodes","test_cp_4_nodes_s", "test_cp_3_nodes"]
# print(cp_surprise_comparison)
# #
# ## RESULT
# ## with the current default parameters, that allow the model to properly run with a lower number of nodes, the model with 4 nodes performs the best in regards to total surprise of node 0
# ## using "standard" as the update type prevents the 5 node models from dropping out, they still perform worse than the 4 node model

# ### comparing the surprises of the test models in the random walk environment
# rw_surprise_comparison = compare_surprise([test_rw_low_p,test_rw_low_p_s,test_rw_high_p, test_rw_high_p_s,test_rw_4_nodes,test_rw_4_nodes_s,test_rw_3_nodes])
# rw_surprise_comparison["Model"] = ["test_rw_low_p","test_rw_low_p_s","test_rw_high_p","test_rw_high_p_s","test_rw_4_nodes","test_rw_4_nodes_s","test_rw_3node"]
# print(rw_surprise_comparison)
# #
# ## RESULT
# ## with the current default parameters, that allow the model to properly run with a lower number of nodes, the model with 4 nodes and "standard" update type performs the best in regards to total surprise of node 0
# ## using "standard" as the update type prevents the 5 node models from dropping out, they still perform worse than the 4 node model
# ## difference between "standard" and default update type very small

# ## comparing the surprise of the 4 node model across environments and update-types
#
surprise_comparison_4n = compare_surprise([test_rw_4_nodes,test_rw_4_nodes_s, test_cp_4_nodes,test_cp_4_nodes_s])
surprise_comparison_4n["Model"]= ["test_rw_4_nodes","test_rw_4_nodes_s", "test_cp_4_nodes","test_cp_4_nodes_s"]
print(surprise_comparison_4n)
#
# ## RESULT
# ## Best performance in random walk environment if update-type is "standard" (6538), if using the default one model performs better in Change point environment (seed 42: 6831 vs. 7047)


# ### comparing the precision trajectories of node 3
#
# rw_precision_comp = compare_trajectories(test_rw_low_p,test_rw_high_p,3,"precision")
# od_precision_comp = compare_precisions(test_od_low_p,test_od_high_p,3,"precision")
#
# rw_precision_comp.to_csv("rw_precision_comp.csv")
# od_precision_comp.to_csv("od_precision_comp.csv")
#
# ## RESULTS
# ## higher precision of node 4 leads to higher precision of node 3 in the oddball environment
# ## opposite is true in the random walk environment

    # precision 3 seems to be the sweet spot so far, such that the first 500 trials can be predicted
    # and are shown in graph (more trials still not working)
    # -> dicotomy between two environments with random walk environment giving up earlier with
    # higher preciscion
    # -> works fine if node 4 is removed

## extracting the node trajectories as data frames
#
# hp_df = test_rw_high_p.to_pandas()
# hp_df = test_od_high_p.to_pandas()
# lp_df = test_rw_low_p.to_pandas()
# lp_df = test_od_low_p.to_pandas()
#
# test_rw_3node_df = test_rw_3node.to_pandas()
# test_od_3node_df = test_od_3node.to_pandas()
#
# test_rw_n4_va_lp_df = test_rw_n4_va_lp.to_pandas()
# test_rw_n4_va_hp_df = test_rw_n4_va_hp.to_pandas()
# test_od_n4_va_lp_df = test_od_n4_va_lp.to_pandas()
# test_od_n4_va_hp_df = test_od_n4_va_hp.to_pandas()


### plotting the trajectories of the different model instances
#
# test_rw_low_p.plot_trajectories()
# test_od_low_p.plot_trajectories()
# test_rw_high_p.plot_trajectories()
# test_od_high_p.plot_trajectories()
# #
# test_rw_3node.plot_trajectories()
# test_od_3node.plot_trajectories()
#
# test_rw_n4_va_lp.plot_trajectories()
# test_rw_n4_va_hp.plot_trajectories()
# test_od_n4_va_lp.plot_trajectories()
# test_od_n4_va_hp.plot_trajectories()


# ## checking the highest jump between observations (before the model cuts off if it does so)
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
#
#
# doing the same with node 4 as a value parent
# print("Highest jumps with node 4 as a value parent (random walk environment): ")
# print(" - Low precision: " + str(test_rw_n4_va_lp.largest_jump()))
# print(" - High precision: " + str(test_rw_n4_va_hp.largest_jump())+ "\n")
#
# print("Highest jumps with node 4 as a value parent (changepoint environment): ")
# print(" - Low precision: " + str(test_od_n4_va_lp.largest_jump()))
# print(" - High precision: " + str(test_od_n4_va_hp.largest_jump())+ "\n")
#
## RESULT:
# highest jump for the high precision rw model is said to be at observation 216 -> does not count the one that actually does it in
# -> oddball at 503 kills the model


## checking weird surprise spike in rw_3node
# print("Highest Surprise: ", test_rw_3node.max_total_surprise())
# test_rw_3node.plot_trajectories()
#
## RESULT
## High surprise at observation 504, because volatility was on a downward trajectory and suddenly jumped up when observation jumped from 269 to 6
## Surprise was less at the higher jump from 487 to 38 at observation 821, as the overall volatility was higher at that point anyway


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
# test_rw_n4_va_hp_df.to_csv("test_rw_n4_va_hp.csv")



# test.plot_trajectories()
# test2.plot_trajectories()
# test3.plot_trajectories()
# test4.plot_trajectories()
# jumps_overview = pd.DataFrame({"Environment": ["Cange Point", "Random Walk"], "mean first jump": [0,0]})
# cp_jumps = []
# rw_jumps = []

# for i in range(1001):
#     current_cp = generate_change_point_environment(n_trials=10, oddball_hazard_rate=0.15, sigma=20, change_point_hazard_rate=0.1,seed=i)["x"].to_numpy()
#     current_rw = generate_random_walk_environment(n_trials=10, oddball_hazard_rate=0.15, sigma=20,seed=i)["x"].to_numpy()
#     current_cp_jump = abs(current_cp[0]-current_cp[1])
#     current_rw_jump = abs(current_rw[0]-current_rw[1])

#     cp_jumps.append(current_cp_jump)
#     rw_jumps.append(current_rw_jump)

# jumps_overview.loc[0, "mean first jump"] = np.mean(cp_jumps)
# jumps_overview.loc[1, "mean first jump"] = np.mean(rw_jumps)

# print(jumps_overview)

# plt.plot(cp_jumps)
# plt.plot(rw_jumps)


# cp_diffs = []
# rw_diffs = []

# for i in range(1001):
#     current_cp = generate_change_point_environment(n_trials=10, oddball_hazard_rate=0.15, sigma=20, change_point_hazard_rate=0.1,seed=i)["x"].to_numpy()
#     current_rw = generate_random_walk_environment(n_trials=10, oddball_hazard_rate=0.15, sigma=20,seed=i)["x"].to_numpy()
#     current_cp_diff = abs(250-current_cp[0])
#     current_rw_diff = abs(250-current_rw[0])

#     cp_diffs.append(current_cp_diff)
#     rw_diffs.append(current_rw_diff)

# diffs_overview = pd.DataFrame({"CP diffs": cp_diffs, "RW diffs": rw_diffs})
# diffs_overview.to_csv("diffs_overview.csv")


## OUTDATED as result was bad with Node 4 as a value parent
#
# test_rw_n4_va_lp = Weber_model(random_walk_data,node_4_type= "value_parent") #model  fit to random walk environment with node 4 as a value parent with comparatively low precision
# test_rw_n4_va_hp = Weber_model(random_walk_data,node_4_type= "value_parent", n4_p= 1e1) #model  fit to random walk environment with node 4 as a value parent with comparatively high precision
#
# test_od_n4_va_lp = Weber_model(oddball_data,node_4_type= "value_parent") #model  fit to random walk environment with node 4 as a value parent with comparatively low precision
# test_od_n4_va_hp = Weber_model(oddball_data,node_4_type= "value_parent", n4_p= 1e1) #model  fit to random walk environment with node 4 as a value parent with comparatively high precision
