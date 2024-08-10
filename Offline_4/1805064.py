import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)
 

def perform_pca(D):
    # Centering the data
    # Standardization
    mean_vector = np.mean(D, axis=0)
    std_dev_vector = np.std(D, axis=0)
    standardized_data = (D - mean_vector) / std_dev_vector

    # Performing Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(standardized_data, full_matrices=False)

    # Projecting the data onto the first two principal components
    reduced_data = np.dot(standardized_data, Vt[:2, :].T)

    return reduced_data

def initialize_parameters(data, k):

    #Random initialization
    indices = np.random.choice(len(data), k, replace=False)
    means = data[indices, :]
    covariances = [np.eye(data.shape[1])] * k
    mixing_coefficients = np.ones(k) / k
    initial_log_likelihood = calculate_log_likelihood(data, means, covariances, mixing_coefficients)

    return means, covariances, mixing_coefficients, initial_log_likelihood

def expectation_maximization(data, k, max_iterations=100, epsilon=1e-6):
    means, covariances, mixing_coefficients, prev_log_likelihood = initialize_parameters(data, k)

    for _ in range(max_iterations):
        # E-step
        responsibilities = e_step(data, means, covariances, mixing_coefficients)
        # M-step
        means, covariances, mixing_coefficients = m_step(data, responsibilities)
        
        current_log_likelihood = calculate_log_likelihood(data, means, covariances, mixing_coefficients)
            
        if np.abs(current_log_likelihood - prev_log_likelihood) < epsilon:
            break
        
        prev_log_likelihood = current_log_likelihood

    return current_log_likelihood, means, covariances, mixing_coefficients

def e_step(data, means, covariances, mixing_coefficients):
    responsibilities = np.zeros((len(data), len(means)))

    for i in range(len(means)):
        responsibilities[:, i] = mixing_coefficients[i] * multivariate_normal_pdf(data, means[i], covariances[i])

    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    return responsibilities

def m_step(data, responsibilities):
    n_k = responsibilities.sum(axis=0)

    means = np.dot(responsibilities.T, data) / n_k[:, np.newaxis]
    covariances = [np.dot((responsibilities[:, i] * (data - means[i]).T), (data - means[i])) / n_k[i] + np.eye(data.shape[1]) * 1e-6 for i in range(len(means))]
    mixing_coefficients = n_k / len(data)

    return means, covariances, mixing_coefficients

def calculate_log_likelihood(data, means, covariances, mixing_coefficients):
    likelihoods = np.zeros((len(data), len(means)))

    for i in range(len(means)):
        likelihoods[:, i] = mixing_coefficients[i] * multivariate_normal_pdf(data, means[i], covariances[i])

    log_likelihood = np.sum(np.log(likelihoods.sum(axis=1)))

    return log_likelihood

def multivariate_normal_pdf(x, mean, covariance):

    if len(x.shape) == 1:
        x = x.reshape(1, -1)  # Convert single data point to a row vector

    k = len(mean)  # Dimension of the multivariate normal distribution
    det_covariance = np.linalg.det(covariance)
    inv_covariance = np.linalg.inv(covariance)
    normalization_factor = 1 / ((2 * np.pi) ** (k / 2) * np.sqrt(det_covariance))

    # Mahalanobis distance
    mahalanobis = np.sum(np.dot((x - mean), inv_covariance) * (x - mean), axis=1)

    # Probability density function
    pdf = normalization_factor * np.exp(-0.5 * mahalanobis)
    return pdf


# Path to the data file
file_path = '100D_data_points.txt'

# Using numpy's genfromtxt function to load the data into a matrix
D = np.genfromtxt(file_path, delimiter=',')


# Perform PCA
if D.shape[1] > 2:
    reduced_data = perform_pca(D)
else:
    reduced_data = D

print("Reduced Data Shape:", reduced_data.shape)
print("Reduced Data Points:")
print(reduced_data)

# Scatter plot of reduced data
sns.set(style="darkgrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Scatter Plot of Reduced Data")
plt.savefig("Scatter_Plot_After_PCA.png")
plt.show()


K_range = range(3, 9)

# Performing Expectation-Maximization for different values of K
best_likelihoods = []
best_models = []

for K in K_range:
    convergence_likelihoods = []

    for _ in range(5):  # Run the algorithm five times for each K
        likelihood, _, _, _ = expectation_maximization(reduced_data, K)
        print("K = {}, Log-Likelihood = {}".format(K, likelihood))
        convergence_likelihoods.append(likelihood)

    best_likelihood = max(convergence_likelihoods)
    print("Best Log-Likelihood = {}".format(best_likelihood))
    best_model_index = convergence_likelihoods.index(best_likelihood)
    current_log_likelyhood, best_model_means, covariances, mixing_coefficients = expectation_maximization(reduced_data, K)

    best_likelihoods.append(best_likelihood)
    best_models.append((K, best_model_means))
    
    # Plotting the best model for each K
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=np.argmax(e_step(reduced_data, best_model_means, covariances, mixing_coefficients), axis=1), cmap='viridis')
    plt.scatter(best_model_means[:, 0], best_model_means[:, 1], c='red', marker='x', label='Cluster Centers')
    plt.title('GMM with K = {}'.format(K))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('GMM_with_K_{}.png'.format(K))
    plt.close()

# Plotting the best log-likelihood vs. K
plt.figure(figsize=(10, 6))
plt.plot(K_range, best_likelihoods, marker='o')
plt.title('Best Log-Likelihood vs. Number of Components (K)')
plt.xlabel('Number of Components (K)')
plt.ylabel('Best Log-Likelihood')
plt.grid(True)
plt.savefig('Best_log_likelihood_vs_K.png')
plt.show()