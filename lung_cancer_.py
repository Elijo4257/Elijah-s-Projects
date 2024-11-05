import os  # For file path operations
import numpy as np  # For array operations
from PIL import Image  # For image processing
import time  # For timing operations
import random as rdm  # Python's built-in random module
from numpy.linalg import svd  # For performing Singular Value Decomposition
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # For model evaluation metrics
import pandas as pd  # For data manipulation using DataFrames
from skimage import color  # For color image processing
from numpy import dstack  # For stacking arrays along the third dimension

# Function to read and resize images
def get_images(path, m, n):
    images = []
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return np.array(images)
    
    print(f"Reading images from: {path}")
    for (dirpath, dirnames, filenames) in os.walk(path):
        for fname in filenames:
            if fname.endswith(".jpeg"):  # Only get jpeg images
                try:
                    # Load the image, convert it to grayscale, and resize
                    img = Image.open(os.path.join(dirpath, fname)).convert('L')
                    img_resized = img.resize((m, n), Image.LANCZOS)
                    img_array = np.array(img_resized).ravel()
                    images.append(img_array)
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
    return np.array(images)  # Return a NumPy array for matrix manipulation

# Lung cancer folder
lung_cancer_dir = r'C:\Users\Elilah\Downloads\Python\lung_colon_image_set\lung_image_sets'
lung_aca_dir = os.path.join(lung_cancer_dir, 'lung_aca')  # Path to lung aca images
lung_n_dir = os.path.join(lung_cancer_dir, 'lung_n')  # Path to lung n images
lung_scc_dir = os.path.join(lung_cancer_dir, 'lung_scc')  # Path to lung scc images

# Reduced size of the images
m = 50
n = 50

# Loading images
print("Loading cancer lung ACA images...")
cancer_lung_aca = get_images(lung_aca_dir, m, n)
print("Loaded cancer lung ACA images:", cancer_lung_aca.shape)

print("Loading cancer lung SCC images...")
cancer_lung_scc = get_images(lung_scc_dir, m, n)
print("Loaded cancer lung SCC images:", cancer_lung_scc.shape)

print("Loading non-cancer lung images...")
no_cancer_lung = get_images(lung_n_dir, m, n)
print("Loaded non-cancer lung images:", no_cancer_lung.shape)

# Define the sizes of the training sets
trainSetSize_lung_aca = 4000
trainSetSize_lung_scc = 4000
trainSetSize_lung_n = 4000

# Lung_aca train and test indices
train_index_lung_aca = np.random.choice(range(5000), trainSetSize_lung_aca, replace=False)
test_index_lung_aca = np.setdiff1d(np.arange(5000), train_index_lung_aca)

# Lung_aca train and test matrices
test_set_lung_aca = np.take(cancer_lung_aca, test_index_lung_aca, axis=0)
train_set_lung_aca = np.take(cancer_lung_aca, train_index_lung_aca, axis=0)

# Lung_scc train and test indices
train_index_lung_scc = np.random.choice(range(5000), trainSetSize_lung_scc, replace=False)
test_index_lung_scc = np.setdiff1d(np.arange(5000), train_index_lung_scc)

# Lung_scc train and test matrices
test_set_lung_scc = np.take(cancer_lung_scc, test_index_lung_scc, axis=0)
train_set_lung_scc = np.take(cancer_lung_scc, train_index_lung_scc, axis=0)

# Lung_n train and test indices
train_index_lung_n = np.random.choice(range(5000), trainSetSize_lung_n, replace=False)
test_index_lung_n = np.setdiff1d(np.arange(5000), train_index_lung_n)

# Lung_n train and test matrices
test_set_lung_n = np.take(no_cancer_lung, test_index_lung_n, axis=0)
train_set_lung_n = np.take(no_cancer_lung, train_index_lung_n, axis=0)

test_sets = [
    (test_set_lung_aca, 'Lung_ACA'),
    (test_set_lung_scc, 'Lung_SCC'),
    (test_set_lung_n, 'Lung_N')
]

# Mean-shifted data for lung
mean_lung_data = np.mean(np.vstack((train_set_lung_aca, train_set_lung_scc, train_set_lung_n)), axis=0)  # Get average looking lung
mean_shifted_lung_aca = train_set_lung_aca - mean_lung_data
mean_shifted_lung_scc = train_set_lung_scc - mean_lung_data
mean_shifted_lung_n = train_set_lung_n - mean_lung_data

class RSVD:
    def __init__(self, rank, p=5):
        self.rank = rank
        self.p = p

    def rsvd(self, X):
        # Perform randomized SVD on the data X
        m, n = X.shape
        P = np.random.randn(n, self.rank + self.p)  # Gaussian random matrix
        Z = X @ P
        Q, _ = np.linalg.qr(Z)
        Y = Q.T @ X
        U_hat, S, Vt = np.linalg.svd(Y, full_matrices=False)
        U = Q @ U_hat
        return U, S, Vt
    
class ImageClassifier:
    def __init__ (self, mean_shifted_lung_aca, mean_shifted_lung_scc, mean_shifted_lung_n,
                 m, n, r, p, q, randomized=True):
        self.m = m
        self.n = n
        self.r = r

         # Reshape the mean image data into rows (1, m * n)
        self.lung_aca_mu = mean_lung_data.reshape(1, m * n)  # 1 x 2500 (50 x 50)
        self.lung_scc_mu = mean_lung_data.reshape(1, m * n)
        self.lung_n_mu = mean_lung_data.reshape(1, m * n)

        if randomized:
            rsvd_instance = RSVD(r, p)
            _, _, self.Vt_lung_aca = rsvd_instance.rsvd(mean_shifted_lung_aca)
            _, _, self.Vt_lung_scc = rsvd_instance.rsvd(mean_shifted_lung_scc)
            _, _, self.Vt_lung_n = rsvd_instance.rsvd(mean_shifted_lung_n)
        else:
            _, _, self.Vt_lung_aca = svd(mean_shifted_lung_aca, full_matrices=False)
            _, _, self.Vt_lung_scc = svd(mean_shifted_lung_scc, full_matrices=False)
            _, _, self.Vt_lung_n = svd(mean_shifted_lung_n, full_matrices=False)
        
        # Ensure Vt arrays are numpy arrays
        self.Vt_lung_aca = np.array(self.Vt_lung_aca)
        self.Vt_lung_scc = np.array(self.Vt_lung_scc)
        self.Vt_lung_n = np.array(self.Vt_lung_n)
        
        # Project the mean-shifted datasets onto the low-dimensional subspace
        self.lung_aca_projection = self.low_rank_representation(mean_shifted_lung_aca, self.Vt_lung_aca.T)
        self.lung_scc_projection = self.low_rank_representation(mean_shifted_lung_scc, self.Vt_lung_scc.T)
        self.lung_n_projection = self.low_rank_representation(mean_shifted_lung_n, self.Vt_lung_n.T)

    def low_rank_representation(self, f, Vt):
        return f @ Vt
        
    def detect_image(self, g):
        g = g.reshape((1, -1)) #reshaping into a 1, m*n
        g_mean_shifted_lung_aca = g - self.lung_aca_mu
        g_mean_shifted_lung_scc = g - self.lung_scc_mu
        g_mean_shifted_lung_n = g - self.lung_n_mu

        g_projected_lung_aca = self.low_rank_representation(g_mean_shifted_lung_aca, self.Vt_lung_aca.T)
        g_projected_lung_scc = self.low_rank_representation(g_mean_shifted_lung_scc, self.Vt_lung_scc.T)
        g_projected_lung_n = self.low_rank_representation(g_mean_shifted_lung_n, self.Vt_lung_n.T)

        distances_lung_aca = np.linalg.norm(self.lung_aca_projection - g_projected_lung_aca, axis=1)
        distances_lung_scc = np.linalg.norm(self.lung_scc_projection - g_projected_lung_scc, axis=1)
        distances_lung_n = np.linalg.norm(self.lung_n_projection - g_projected_lung_n, axis=1)

        min_distance_lung_aca = np.min(distances_lung_aca)
        min_distance_lung_scc = np.min(distances_lung_scc)
        min_distance_lung_n = np.min(distances_lung_n)

        if min_distance_lung_scc < min_distance_lung_aca and min_distance_lung_scc < min_distance_lung_n:
            result_category = 'Lung_SCC'
            min_distance = min_distance_lung_scc
        elif min_distance_lung_aca < min_distance_lung_n and min_distance_lung_aca < min_distance_lung_scc:
            result_category = 'Lung_ACA'
            min_distance = min_distance_lung_aca
        else:
            result_category = 'Lung_N'
            min_distance = min_distance_lung_n

        return result_category, min_distance

m, n, r, p, q = 50, 50, 3, 5, 3  

def evaluate_classifier(classifier, test_sets):
    # Combine all test sets and labels
    all_test_images = []
    all_labels = []

    for test_set, label in test_sets:
        all_test_images.extend(test_set)
        all_labels.extend([label] * len(test_set))

    # Convert to numpy arrays for easier manipulation
    all_test_images = np.array(all_test_images)
    all_labels = np.array(all_labels)

    # Run the classifier on the test set
    predicted_labels = []
    total_time = 0

    for random_test_image in all_test_images:
        start_time = time.time()
        result_category, _ = classifier.detect_image(random_test_image)
        end_time = time.time()
        
        predicted_labels.append(result_category)
        total_time += (end_time - start_time)

    predicted_labels = np.array(predicted_labels)

    # Calculate accuracy
    accuracy = np.mean(predicted_labels == all_labels)

    # Calculate average time per classification
    average_time_per_classification = total_time / len(all_test_images)

    return accuracy, average_time_per_classification

# Evaluate with Randomized SVD
start_time = time.time()
classifier_randomized = ImageClassifier(mean_shifted_lung_aca, mean_shifted_lung_scc, mean_shifted_lung_n, m, n, r, p, q, randomized=True)
accuracy_randomized, time_randomized = evaluate_classifier(classifier_randomized, test_sets)
end_time = time.time()
print(f"Randomized SVD: Accuracy = {accuracy_randomized*100:.2f}%, Average Time per Classification = {time_randomized:.4f} seconds, Total Time = {end_time - start_time:.2f} seconds")

# Evaluate with Regular SVD
start_time = time.time()
classifier_regular = ImageClassifier(mean_shifted_lung_aca, mean_shifted_lung_scc, mean_shifted_lung_n, m, n, r, p, q, randomized=False)
accuracy_regular, time_regular = evaluate_classifier(classifier_regular, test_sets)
end_time = time.time()
print(f"Regular SVD: Accuracy = {accuracy_regular*100:.2f}%, Average Time per Classification = {time_regular:.4f} seconds, Total Time = {end_time - start_time:.2f} seconds")
