import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment
from cognitive_oddballs.models.weber_model import WeberModel as Weber_model


# Author: Lucy Heuer

def compare_trajectories(models: list, node_idx: int, col_to_compare: str) -> pd.DataFrame:
    """Compiles the wanted node_trajectories into a DataFrame to ease comparison.
    All Models must have the node to be compared

    Input:
    - models: a list containing the models to be compared
    - node_idx: Index of the node to be compared (must be present in all models)
    - col_to_compare: Possible values: "expected_mean","expected_precision","mean","precision","surprise"

    Output:
    - DataFrame containing the values in question"""

    accepted_columns = ["expected_mean", "expected_precision", "mean", "precision", "surprise"]
    if col_to_compare not in accepted_columns:
        raise ValueError(
            "col_to_compare must be one of the following: 'expected_mean','expected_precision','mean','precision','surprise'"
        )

    output = pd.DataFrame()
    nth_model = 1

    for model in models:
        current_df = model.to_pandas()
        if len(current_df.columns) < ((node_idx + 1) * 6 + 4):
            raise ValueError("Node with given Index must be present in all Models.")

        model_col_name = str("x_" + str(node_idx) + "_" + col_to_compare)
        df_col_name = str(
            "Model " + str(nth_model) + " Node " + str(node_idx) + " " + col_to_compare
        )
        output[df_col_name] = current_df[model_col_name]
        nth_model += 1

    return output


def compare_surprise(models: list, model_names: list) -> pd.DataFrame:
    """Compares the over all surprise of node 0 of the given Models

    Input:
    - models: A list of models to be compared
    - model_names: A list containing the name identifying each model

    Output:
    - DataFrame containing the total surprises of node 0 for the models, the max surprise for each, whether it cuts of and which one has the lowest total surprise (excluding models which cut off)"""

    comparison = pd.DataFrame(
        {
            "Model": model_names,
            "Total_Surprise": [1.1] * len(models),
            "Max_Surprise": [1.1] * len(models),
            "Cuts_off": [False] * len(models),
            "Has_lowest_surprise": [False] * len(models),
        }
    )
    for i in range(len(models)):
        current_model_df = models[i].to_pandas()

        comparison.loc[i, "Total_Surprise"] = sum(current_model_df["x_0_surprise"])
        comparison.loc[i, "Max_Surprise"] = max(current_model_df["x_0_surprise"])
        comparison.loc[i, "Cuts_off"] = models[i].drops_out()

    lowest_surprise = np.nanmax(comparison["Total_Surprise"])
    lowest_model = len(models)
    for k in range(len(models)):
        if (
            comparison.loc[k, "Total_Surprise"] <= lowest_surprise
            and not comparison.loc[k, "Cuts_off"]
        ):
            lowest_surprise = comparison.loc[k, "Total_Surprise"]
            lowest_model = k
    comparison.loc[lowest_model, "Has_lowest_surprise"] = True

    return comparison


def compare_highest_jump(models: list, model_names: list) -> pd.DataFrame:
    """Compares the highest jump between observations and whether the model dropped out

    Input:
    - models: A list of models to be compared
    - model_names: A list containing the name identifying each model

    Output:
    -  A DataFrame containing the highest jump between observations the model percieves + index (before it drops out) and whether it does so"""

    comparison = pd.DataFrame(
        {
            "Model": model_names,
            "Highest Jump": [1.1] * len(models),
            "At": [1.1] * len(models),
            "Cuts off": [False] * len(models),
        }
    )
    for i in range(len(models)):
        highest_jump, at = models[i].largest_jump()

        comparison.loc[i, "Highest Jump"] = highest_jump
        comparison.loc[i, "At"] = at
        comparison.loc[i, "Cuts off"] = models[i].drops_out()

    return comparison


def comparing_mean_jumps(m1: Weber_model, m2: Weber_model, model_names: list) -> pd.DataFrame:
    """Compares the node 0 means of two given models and the jumps in those trajectories.
    Input:
    - m1: The first model to be compared
    - m2: The second model to be compared
    - model_names: A list of strings naming the compared models

    Output:
    - prints information about the objectively largest jump (if all means are the same) and what the largest jumps the models actually perceived are.
    - comparison: A pandas Dataframe containing the means of the given models, whether they are the same and jumps between the observed means.

    If all means are the same there is only one "Jumps" collumn, else there is a "Jumps" collumn for each model respectively
    """
    comparison = compare_trajectories([m1, m2], 0, "mean")
    comparison["Is same"] = comparison.iloc[:, 0] == comparison.iloc[:, 1]
    if len(comparison) == sum(comparison["Is same"]):
        comparison["Jumps"] = range(len(comparison))
        for i in range(1, len(comparison)):
            comparison.loc[i, "Jumps"] = abs(comparison.iloc[i, 0] - comparison.iloc[i - 1, 0])

        print("Objective max. Jump in Observations: " + str(max(comparison["Jumps"])))
    else:
        m1_col_name = model_names[0] + " Jumps"
        m2_col_name = model_names[1] + " Jumps"
        comparison[m1_col_name] = range(len(comparison))
        comparison[m2_col_name] = range(len(comparison))

        for j in range(1, len(comparison)):
            comparison.loc[j, m1_col_name] = abs(comparison.iloc[i, 0] - comparison.iloc[i - 1, 0])
            comparison.loc[j, m2_col_name] = abs(comparison.iloc[i, 1] - comparison.iloc[i - 1, 1])

    m1_max_jump, m1_at = m1.largest_jump()
    m2_max_jump, m2_at = m2.largest_jump()
    print(
        "Subjective max. Jumps: \n - "
        + model_names[0]
        + ": "
        + str(m1_max_jump)
        + " at index "
        + str(m1_at)
        + "\n - "
        + model_names[1]
        + ": "
        + str(m2_max_jump)
        + " at index "
        + str(m2_at)
    )

    return comparison

#################################################################################################################
# The code in this file is intended to test the functioning of the Model based on Weber et al. 2023
# It somewhat chronicles the development of our model structure and some parts might seem redundant/slightly inconsistent because of that

# The code in general is intended to be commented/ uncommented as needed 
# (That is why large chunks of code are currently commented)
#########################################################################################



### trying to find starting values, that make the model run according to Prof. Webers suggestions

## comparing the mean first jumps across many instantiations of both environments

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


### looking at the difference between the first observation and 250 for many instantiations of the environments
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



# ## the default parameters currently implemented for Weber_model are the ones allowing the model to run with only 3 nodes (as Weber suggested)

# # # ## generating data from both environments
change_point_data = generate_change_point_environment(
    n_trials=1000, oddball_hazard_rate=0.15, sigma=20, change_point_hazard_rate=0.1, seed=42
)
random_walk_data = generate_random_walk_environment(
    n_trials=1000, oddball_hazard_rate=0.15, sigma=20, seed=42
)


# ### creating Model instances with different parameters

# ## 5 nodes (different precisions) with update_type="eHGF"
# test_rw_low_p = Weber_model(n_nodes=5, x_4_p=3,update_type="eHGF").input_data(random_walk_data["x"].to_numpy())
# test_cp_low_p = Weber_model(n_nodes=5, x_4_p=3,update_type="eHGF").input_data(change_point_data["x"].to_numpy())
# test_rw_high_p = Weber_model(n_nodes=5,update_type="eHGF").input_data(random_walk_data["x"].to_numpy())
# test_cp_high_p = Weber_model(n_nodes=5,update_type="eHGF").input_data(change_point_data["x"].to_numpy())

# ## 5 nodes with "standard" as the update type
# test_rw_low_p_s = Weber_model(n_nodes=5, x_4_p=3, update_type="standard").input_data(
#     random_walk_data["x"].to_numpy()
# )
# test_cp_low_p_s = Weber_model(n_nodes=5, x_4_p=3, update_type="standard").input_data(
#     change_point_data["x"].to_numpy()
# )
test_rw_high_p_s = Weber_model(n_nodes=5, update_type="standard").input_data(
    random_walk_data["x"].to_numpy()
)
test_cp_high_p_s = Weber_model(n_nodes=5, update_type="standard").input_data(
    change_point_data["x"].to_numpy()
)

# ## 4 nodes with update_type="eHGF"
test_rw_4_nodes = Weber_model(n_nodes=4,update_type="eHGF").input_data(random_walk_data["x"].to_numpy())
test_cp_4_nodes = Weber_model(n_nodes=4,update_type="eHGF").input_data(change_point_data["x"].to_numpy())

# ## 4 nodes with "standard" as the update type
test_rw_4_nodes_s = Weber_model(n_nodes=4, update_type="standard").input_data(
    random_walk_data["x"].to_numpy()
)
test_cp_4_nodes_s = Weber_model(n_nodes=4, update_type="standard").input_data(
    change_point_data["x"].to_numpy()
)

# # ## 3 nodes  with update_type="eHGF"
# test_rw_3_nodes = Weber_model(n_nodes=3,update_type="eHGF").input_data(random_walk_data["x"].to_numpy())
# test_cp_3_nodes = Weber_model(n_nodes=3,update_type="eHGF").input_data(change_point_data["x"].to_numpy())

# # ##3 nodes with "standard" as update type
# test_rw_3_nodes_s = Weber_model(n_nodes=3, update_type="standard").input_data(
#     random_walk_data["x"].to_numpy()
# )
# test_cp_3_nodes_s = Weber_model(n_nodes=3, update_type="standard").input_data(
#     change_point_data["x"].to_numpy()
# )

### comparing the surprises of the test models in the change point environment
# print("comparing the surprises of the test models in the change point environment")
# cp_surprise_comparison = compare_surprise(
#     [
#         test_cp_high_p,
#         test_cp_high_p_s,
#         test_cp_low_p,
#         test_cp_low_p_s,
#         test_cp_4_nodes,
#         test_cp_4_nodes_s,
#         test_cp_3_nodes,
#         test_cp_3_nodes_s,
#     ],
#     [
#         "test_cp_high_p",
#         "test_cp_high_p_s",
#         "test_cp_low_p",
#         "test_cp_low_p_s",
#         "test_cp_4_nodes",
#         "test_cp_4_nodes_s",
#         "test_cp_3_nodes",
#         "test_cp_3_nodes_s",
#     ],
# )
# print(cp_surprise_comparison)
#
## RESULT
## with the current default parameters, that allow the model to properly run with a lower number of nodes, the model with 4 nodes performs the best in regards to total surprise of node 0
## using "standard" as the update type prevents the 5 node models from dropping out, they still perform worse than the 4 node model

# ### comparing the surprises of the test models in the random walk environment
# print("comparing the surprises of the test models in the random walk environment")
# rw_surprise_comparison = compare_surprise(
#     [
#         test_rw_low_p,
#         test_rw_low_p_s,
#         test_rw_high_p,
#         test_rw_high_p_s,
#         test_rw_4_nodes,
#         test_rw_4_nodes_s,
#         test_rw_3_nodes,
#         test_rw_3_nodes_s,
#     ],
#     [
#         "test_rw_low_p",
#         "test_rw_low_p_s",
#         "test_rw_high_p",
#         "test_rw_high_p_s",
#         "test_rw_4_nodes",
#         "test_rw_4_nodes_s",
#         "test_rw_3node",
#         "test_rw_3_nodes_s",
#     ],
# )
# print(rw_surprise_comparison)
# #
# ## RESULT
# ## with the current default parameters, that allow the model to properly run with a lower number of nodes, the model with 4 nodes and "standard" update type performs the best in regards to total surprise of node 0
# ## using "standard" as the update type prevents the 5 node models from dropping out, they still perform worse than the 4 node model
# ## difference between "standard" and default update type very small

# ## comparing the surprise of the 4 node model across environments and update-types
#
# print("comparing the surprise of the 4 node model across environments and update-types")
# surprise_comparison_4n = compare_surprise(
#     [test_rw_4_nodes, test_rw_4_nodes_s, test_cp_4_nodes, test_cp_4_nodes_s],
#     ["test_rw_4_nodes", "test_rw_4_nodes_s", "test_cp_4_nodes", "test_cp_4_nodes_s"],
# )
# print(surprise_comparison_4n)
#
# ## RESULT
# ## Best performance in random walk environment if update-type is "standard" (6538), if using the default one model performs better in Change point environment (seed 42: 6831 vs. 7047)
# ## With "standard" update type the difference between performance in cp and rw environments is not very big (6566 vs. 6538)


# ## extracting the node trajectories as data frames

# # 5 nodes (different precisions)
# rw_low_p_df = test_rw_low_p.to_pandas()
# cp_low_p_df = test_cp_low_p.to_pandas()
# rw_high_p_df = test_rw_high_p.to_pandas()
# cp_high_p_df = test_cp_high_p.to_pandas()

# ## 5 nodes with "standard" as the update type
# rw_low_p_s_df = test_rw_low_p_s.to_pandas()
# cp_low_p_s_df = test_cp_low_p_s.to_pandas()
# rw_high_p_s_df = test_rw_high_p_s.to_pandas()
# cp_high_p_s_df = test_cp_high_p_s.to_pandas()

# ## 4 nodes
# rw_4_nodes_df = test_rw_4_nodes.to_pandas()
# cp_4_nodes_df = test_cp_4_nodes.to_pandas()

# ## 4 nodes with "standard" as the update type
# rw_4_nodes_s_df = test_rw_4_nodes_s.to_pandas()
# cp_4_nodes_s_df = test_cp_4_nodes_s.to_pandas()

# ## 3 nodes
# rw_3_nodes_df = test_rw_3_nodes.to_pandas()
# cp_3_nodes_df = test_cp_3_nodes.to_pandas()

# ## 3 nodes with "standard" as the update type
# rw_3_nodes_s_df = test_rw_3_nodes_s.to_pandas()
# cp_3_nodes_s_df = test_cp_3_nodes_s.to_pandas()


# ### plotting the trajectories of the different model instances
# # 5 nodes (different precisions)
# test_rw_low_p.plot_trajectories()
# test_cp_low_p.plot_trajectories()
# test_rw_high_p.plot_trajectories()
# test_cp_high_p.plot_trajectories()

# ## 5 nodes with "standard" as the update type
# test_rw_low_p_s.plot_trajectories()
# test_cp_low_p_s.plot_trajectories()
test_rw_high_p_s.plot_trajectories()
test_cp_high_p_s.plot_trajectories()

# ## 4 nodes
test_rw_4_nodes.plot_trajectories()
test_cp_4_nodes.plot_trajectories()

# ## 4 nodes with "standard" as the update type
test_rw_4_nodes_s.plot_trajectories()
test_cp_4_nodes_s.plot_trajectories()

# ## 3 nodes
# test_rw_3_nodes.plot_trajectories()
# test_cp_3_nodes.plot_trajectories()

# ## 3 nodes with "standard" as the update type
# test_rw_3_nodes_s.plot_trajectories()
# test_cp_3_nodes_s.plot_trajectories()


# ## comparing the highest jump between observations the models percieves before dropping out in the different environments and between different precision values for node 4
## with 5 nodes
# jump_comparison_5_nodes = compare_highest_jump([test_rw_low_p, test_rw_high_p, test_cp_low_p, test_cp_high_p], ["test_rw_low_p", "test_rw_high_p", "test_cp_low_p", "test_cp_high_p"])
# print(jump_comparison_5_nodes)
#
## with 4 nodes
# jump_comparison_4_nodes = compare_highest_jump([test_rw_4_nodes, test_cp_4_nodes],["test_rw_4_nodes", "test_cp_4_nodes"])
# print(jump_comparison_4_nodes)
#
# # RESULT
# # high precision of node 4 leads the model to drop out at a smaller jump (/earlier) in the randomwalk environment (as compared to lower precision of node 4)
# # while high precision of node 4 leads the model to persevere after a overall larger jump in the oddball environment


## checking weird surprise spike in rw_4_nodes
# print("Highest Surprise: ", test_rw_4_nodes.max_total_surprise())
# test_rw_4_nodes.plot_trajectories()
#
## RESULT
## High surprise at observation 504, because volatility was on a downward trajectory and suddenly jumped up when observation jumped from 269 to 6
## Surprise was less at the higher jump from 487 to 38 at observation 821, as the overall volatility was higher at that point anyway


### comparing node 0 mean trajectories and jumps

# # between 5 node model with low precision and 4 node model
# mean_and_jump_comp_5_4 = comparing_mean_jumps(test_rw_low_p, test_rw_4_nodes, ["5 node model with low precision", "4 node model"])
# # saving to csv to ease inspection
# mean_and_jump_comp_5_4.to_csv("5_4_jump_comp.csv")

# # between 5 node models with low and high precision
# mean_and_jump_comp_hp_lp = comparing_mean_jumps(test_rw_low_p,test_rw_high_p, ["Low precision", "High precision"])
# # saving to csv to ease inspection
# mean_and_jump_comp_hp_lp.to_csv("hp_lp_jump_comp.csv")


# ### comparing performance of 5 node network with "standard", a 4 node network with "standard" and 4 node with "eHGF" across many instantiations of environments (500 trials each)
rw_5_surprises=[]
rw_s_surprises=[]
rw_e_surprises=[]

cp_5_surprises=[]
cp_s_surprises=[]
cp_e_surprises=[]

for i in range(100):
    current_cp = generate_change_point_environment(n_trials=500, oddball_hazard_rate=0.15, sigma=20, change_point_hazard_rate=0.1,seed=i)["x"].to_numpy()
    current_rw = generate_random_walk_environment(n_trials=500, oddball_hazard_rate=0.15, sigma=20,seed=i)["x"].to_numpy()

    rw_5_model = Weber_model(n_nodes=5).input_data(current_rw)
    rw_s_model = Weber_model().input_data(current_rw)
    rw_e_model = Weber_model(update_type="eHGF").input_data(current_rw)

    cp_5_model = Weber_model(n_nodes=5).input_data(current_cp)
    cp_s_model = Weber_model().input_data(current_cp)
    cp_e_model = Weber_model(update_type="eHGF").input_data(current_cp)

    comp = compare_surprise(
        [rw_5_model, rw_s_model, rw_e_model, cp_5_model, cp_s_model, cp_e_model],
        ["rw_5_model","rw_s_model","rw_e_model", "cp_5_model", "cp_s_model", "cp_e_model"]
    )
    rw_5_surprises.append(comp.iloc[0,1])
    rw_s_surprises.append(comp.iloc[1,1])
    rw_e_surprises.append(comp.iloc[2,1])
    cp_5_surprises.append(comp.iloc[3,1])
    cp_s_surprises.append(comp.iloc[4,1])
    cp_e_surprises.append(comp.iloc[5,1])

surprise_comparison_across_instances = pd.DataFrame({"rw_5": rw_5_surprises,"rw_s":rw_s_surprises, "rw_e":rw_e_surprises,"cp_5":cp_5_surprises ,"cp_s": cp_s_surprises, "cp_e": cp_e_surprises})
surprise_comparison_across_instances.to_csv("s_c_a_i.csv")
# ## RESULT
# # although removing node 5 lead to better performance initially, there are still cases in which the model drops out. 
# # while the "standard" update type seems to prevent drop outs, this is no quarantee that it could not happen in even more volatile environments.

### Directly comparing the performance of the 5 node model with "standard" update type to the 4 node model of the same type

cmap = plt.get_cmap("Blues")
colors = cmap(np.linspace(0,1,6))
fig,ax = plt.subplots()
ax.set_ylabel("Summed surprise of node 0")

bplot = ax.boxplot(
    [rw_5_model, rw_s_model, cp_5_model, cp_s_model],
    tick_labels=["RW 5 nodes", "RW 4 nodes", "CP 5 nodes","CP 4 nodes"],
    patch_artist=True,
    showfliers=False
)
for patch,color in zip(bplot["boxes"],colors):
    patch.set_facecolor(color)

plt.show()


# ### saving stuff trajectories into csvs to ease inspection
# #
# # 5 nodes (different precisions)
# test_rw_low_p.to_pandas().to_csv("test_rw_low_p.csv")
# test_cp_low_p.to_pandas().to_csv("test_cp_low_p.csv")
# test_rw_high_p.to_pandas().to_csv("test_rw_high_p.csv")
# test_cp_high_p.to_pandas().to_csv("test_cp_high_p.csv")

# ## 5 nodes with "standard" as the update type
# test_rw_low_p_s.to_pandas().to_csv("test_rw_low_p_s.csv")
# test_cp_low_p_s.to_pandas().to_csv("test_cp_low_p_s.csv")
# test_rw_high_p_s.to_pandas().to_csv("test_rw_high_p_s.csv")
# test_cp_high_p_s.to_pandas().to_csv("test_cp_high_p_s.csv")

# ## 4 nodes
# test_rw_4_nodes.to_pandas().to_csv("test_rw_4_nodes.csv")
# test_cp_4_nodes.to_pandas().to_csv("test_cp_4_nodes.csv")

# ## 4 nodes with "standard" as the update type
# test_rw_4_nodes_s.to_pandas().to_csv("test_rw_4_nodes_s.csv")
# test_cp_4_nodes_s.to_pandas().to_csv("test_cp_4_nodes_s.csv")

# ## 3 nodes
# test_rw_3_nodes.to_pandas().to_csv("test_rw_3_nodes.csv")
# test_cp_3_nodes.to_pandas().to_csv("test_cp_3_nodes.csv")

# ## 3 nodes with "standard" as the update type
# test_rw_3_nodes_s.to_pandas().to_csv("test_rw_3_nodes_s.csv")
# test_cp_3_nodes_s.to_pandas().to_csv("test_cp_3_nodes_s.csv")


# ------------------------------------------OUTDATED----------------------------------------------------#

## OUTDATED as result was bad with Node 4 as a value parent
#
# test_rw_n4_va_lp = Weber_model(random_walk_data,node_4_type= "value_parent") #model  fit to random walk environment with node 4 as a value parent with comparatively low precision
# test_rw_n4_va_hp = Weber_model(random_walk_data,node_4_type= "value_parent", n4_p= 1e1) #model  fit to random walk environment with node 4 as a value parent with comparatively high precision
#
# test_od_n4_va_lp = Weber_model(oddball_data,node_4_type= "value_parent") #model  fit to random walk environment with node 4 as a value parent with comparatively low precision
# test_od_n4_va_hp = Weber_model(oddball_data,node_4_type= "value_parent", n4_p= 1e1) #model  fit to random walk environment with node 4 as a value parent with comparatively high precision
#
# comparing highest jumps with node 4 as a value parent
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

## copy template

## 5 nodes (different precisions)
# test_rw_low_p
# test_cp_low_p
# test_rw_high_p
# test_cp_high_p

# ## 5 nodes with "standard" as the update type
# test_rw_low_p_s
# test_cp_low_p_s
# test_rw_high_p_s
# test_cp_high_p_s

# ## 4 nodes
# test_rw_4_nodes
# test_cp_4_nodes

# ## 4 nodes with "standard" as the update type
# test_rw_4_nodes_s
# test_cp_4_nodes_s

# ## 3 nodes
# test_rw_3_nodes
# test_cp_3_nodes

# ## 3 nodes with "standard" as the update type
# test_rw_3_nodes_s
# test_cp_3_nodes_s
