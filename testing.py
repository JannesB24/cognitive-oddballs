from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment
from cognitive_oddballs.models.weber_model import WeberModel as Weber_model
import pandas as pd
import numpy as np

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


    comparison = pd.DataFrame({"Model": range(len(models)), "Total_Surprise":range(len(models)), "Max_Surprise": range(len(models)), "Cuts_off":range(len(models)) , "Has_lowest_surprise":([False]*len(models))})
    for i in range(len(models)):
        current_model_df = models[i].to_pandas()
                      
        comparison.loc[i, "Model"] = ("Model "+str(i+1))
        comparison.loc[i, "Total_Surprise"] = sum(current_model_df["x_0_surprise"])
        comparison.loc[i, "Max_Surprise"] = max(current_model_df["x_0_surprise"])
        comparison.loc[i, "Cuts_off"] = (np.count_nonzero(np.isnan(current_model_df[["x_0_surprise"]]))>0)

    lowest_surprise = np.nanmax(comparison["Total_Surprise"])
    lowest_model = len(models)
    for k in range(len(models)):
        if comparison.loc[k,"Total_Surprise"] <= lowest_surprise and not comparison.loc[k, "Cuts_off"]:
            lowest_surprise = comparison.loc[k,"Total_Surprise"]
            lowest_model = k
    comparison.loc[lowest_model,"Has_lowest_surprise"] = True

    return comparison


# ## generating data from both environments
oddball_data = generate_change_point_environment(n_trials=1000, oddball_hazard_rate=0.15, sigma=20, change_point_hazard_rate=0.1, seed=42)
random_walk_data = generate_random_walk_environment(n_trials=1000, oddball_hazard_rate=0.15, sigma=20, seed=42)

test = Weber_model().input_data(random_walk_data["x"].to_numpy())

test2 = Weber_model(True).input_data(random_walk_data["x"].to_numpy())

test3 = Weber_model().input_data(oddball_data["x"].to_numpy())

test4 = Weber_model(True).input_data(oddball_data["x"].to_numpy())

test.plot_trajectories()
test2.plot_trajectories()
test3.plot_trajectories()
test4.plot_trajectories()