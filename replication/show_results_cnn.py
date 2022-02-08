from functions import *

results_df = pd.read_pickle("results.pkl")

print(results_df)

cnn_results = results_df.groupby(results_df["model"]).agg([np.mean, np.std])

cnn_results.to_csv("results_summary.csv")

print(cnn_results)
