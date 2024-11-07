import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import VECM
from pgmpy.estimators import HillClimbSearch, K2Score
from pgmpy.models import BayesianNetwork
from scipy.stats import t


OUTPUT_IMAGE_PATH = "./app/static/images"

# Load the dataset
data = pd.read_csv('./app/dataset.csv')
data = data.set_index('Date')  # Set the date column as index for time series analysis
data = data.dropna()


def adf_test(series, signif=0.05):
    """Perform ADF test and return results as a dictionary."""
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    test_result = {
        'ADF Statistic': result[0],
        'p-value': p_value,
        'Stationarity': "Stationary" if p_value <= signif else "Non-stationary"
    }
    return test_result

results_original = []


# Apply ADF test to each interest rate column in the original series
for column in data.columns:
    result = adf_test(data[column])
    result['Column'] = column  # Add the column name to the result
    results_original.append(result)
results_original_df = pd.DataFrame(results_original)


def get_adf_result():
    return results_original_df


results_diff = []

# Apply ADF test to each interest rate column after differencing
for column in data.columns:
    differenced_series = data[column].diff().dropna()
    result = adf_test(differenced_series)
    result['Column'] = column
    results_diff.append(result)
results_diff_df = pd.DataFrame(results_diff)


def get_adf_after_diff():
    return results_diff_df


# Determine optimal lag order
model = VAR(data)
lag_order = model.select_order(maxlags=10)


def get_lag_order_summary():
    return lag_order.summary()


# Select lag based on SIC (or adjust based on preferred criteria)
optimal_lag = lag_order.selected_orders['bic']  # BIC is the same as SIC


def get_optimal_lag():
    return optimal_lag


# Johansen Cointegration Test
johansen_test = coint_johansen(data, det_order=0, k_ar_diff=optimal_lag)
# Retrieve trace statistics and critical values
trace_stat = johansen_test.lr1  # Trace statistics
crit_values = johansen_test.cvt[:, 1]  # 5% critical values for the trace test


def get_cointegration_results():
    ret = {}
    for i, (trace, crit_val) in enumerate(zip(trace_stat, crit_values), start=1):
        ret[i] = {
            'trace_statisic': trace,
            'critical_value': crit_val
        }
    return ret


# Estimate VECM with cointegration rank 4 and optimal lag 1
vecm_model = VECM(data, k_ar_diff=optimal_lag, coint_rank=3)  # Setting rank to 4
vecm_fit = vecm_model.fit()


def get_vecm_summary():
    return vecm_fit.summary()


# Extract residuals from the fitted VECM model
vecm_residuals = vecm_fit.resid
vecm_residuals_df = pd.DataFrame(vecm_residuals, columns=data.columns)


def get_vecm_residuals():
    return vecm_residuals_df


residuals_corr_matrix = vecm_residuals_df.corr()


def get_correlation_matrix():
    return residuals_corr_matrix


vecm_residuals_df = vecm_residuals_df.dropna(axis=1, how='all')
vecm_residuals_df = vecm_residuals_df.dropna(axis=0, how='all')
vecm_residuals_df = vecm_residuals_df.apply(pd.to_numeric, errors='coerce')

search = HillClimbSearch(residuals_corr_matrix)
model = search.estimate(scoring_method=K2Score(residuals_corr_matrix))


def get_dag():
    graph = nx.DiGraph(model.edges())
    plt.figure(figsize=(22, 9))
    nx.draw_networkx(graph, with_labels=True, node_color="skyblue", node_size=2000, font_size=10)
    plt.title("Initial Bayesian Network Structure")
    output_file = f"{OUTPUT_IMAGE_PATH}/dag.jpg"
    plt.savefig(output_file)

    return "dag.jpg"


# Example correlation matrix
corr_matrix = np.array([
    [1.0, -0.029, 0.065, 0.29, -0.013, -0.0081, 0.14, 0.25],
    [-0.029, 1.0, 0.074, 0.18, 0.13, 0.11, 0.0063, 0.41],
    [0.065, 0.074, 1.0, -0.026, 0.056, -0.014, 0.063, 0.19],
    [0.29, 0.18, -0.026, 1.0, 0.075, 0.06, 0.08, 0.34],
    [-0.013, 0.13, 0.056, 0.075, 1.0, -0.055, -0.067, 0.018],
    [-0.0081, 0.11, -0.014, 0.06, -0.055, 1.0, 0.082, 0.11],
    [0.14, 0.0063, 0.063, 0.08, -0.067, 0.082, 1.0, 0.028],
    [0.25, 0.41, 0.19, 0.34, 0.018, 0.11, 0.028, 1.0]
])

# Define node names corresponding to the matrix rows/columns
nodes = ['Repo Rate', 'Certificate of Deposit', 'Commercial Paper', '365-Bill Yield', 'Loan', 'Term Deposit (>1 year)', 'Saving Deposit', 'Call Rate']

# Create a NetworkX graph for the DAG
G = nx.DiGraph()

# Adding edges to the DAG based on correlation threshold (use a suitable threshold for your data)
threshold = 0.1

for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        if abs(corr_matrix[i, j]) > threshold:
            G.add_edge(nodes[i], nodes[j], weight=corr_matrix[i, j])


def conditional_corr(corr_matrix, i, j, k):
    """Function to calculate conditional correlation coefficients"""
    # Using the formula from the provided equation (7)
    r_ij = corr_matrix[i, j]
    r_ik = corr_matrix[i, k]
    r_jk = corr_matrix[j, k]
    return (r_jk - r_ij * r_ik) / np.sqrt((1 - r_ij ** 2) * (1 - r_ik ** 2))


# Calculate conditional correlations for example nodes
cond_corr_example = conditional_corr(corr_matrix, 0, 1, 2)  # Between nodes 0 and 1 given 2


def dag_conditional_correlation():
    def conditional_corr(Xi, Xj, Xk, corr_matrix):
        """Function to calculate conditional correlation using Equation 7"""
        rho_ij = corr_matrix.loc[Xi, Xj]
        rho_ik = corr_matrix.loc[Xi, Xk]
        rho_jk = corr_matrix.loc[Xj, Xk]
        numerator = rho_jk - (rho_ij * rho_ik)
        denominator = np.sqrt((1 - rho_ij ** 2) * (1 - rho_ik ** 2))
        return numerator / denominator if denominator != 0 else np.nan

    # Threshold for determining significant conditional correlations
    cond_corr_threshold = 0.3
    initial_edges = []

    # Calculate conditional correlations for all unique triplets
    nodes = residuals_corr_matrix.columns

    for Xi in nodes:
        for Xj in nodes:
            if Xi == Xj:
                continue
            for Xk in nodes:
                if Xk in [Xi, Xj]:
                    continue

                # Calculate conditional correlation
                cond_corr = conditional_corr(Xi, Xj, Xk, residuals_corr_matrix)
                # Add edge if conditional correlation exceeds threshold
                if not np.isnan(cond_corr) and abs(cond_corr) > cond_corr_threshold:
                    initial_edges.append((Xi, Xj, cond_corr))  # Xi -> Xj given Xk

    G = nx.DiGraph()

    for edge in initial_edges:
        Xi, Xj, cond_corr = edge
        G.add_edge(Xi, Xj, weight=cond_corr)

    # Draw the initial DAG based on conditional correlations
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    # Adding weights (conditional correlation values) to edges
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Optimised DAG Based on Conditional Correlations")
    output_file = f"{OUTPUT_IMAGE_PATH}/dag_cond_corr.jpg"
    plt.savefig(output_file)
    return "dag_cond_corr.jpg"