import os
import numpy as np
from PIL import Image
import numpy.linalg as la
import random as rdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from numpy.linalg import svd
import pandas as pd
from skimage import color
from numpy import linalg as la 
from numpy import dstack
from numpy import random as rdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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

# Colon cancer folder
colon_cancer_dir = r'C:\Users\Elilah\Downloads\Python\lung_colon_image_set\colon_image_sets'
colon_aca_dir = os.path.join(colon_cancer_dir, 'colon_aca')  # Path to colon aca images
colon_n_dir = os.path.join(colon_cancer_dir, 'colon_n')  # Path to colon n images

# Reduced size of the images
m = 50
n = 50

# Loading images
print("Loading colon ACA images...")
cancer_colon_aca = get_images(colon_aca_dir, m, n)
print("Loaded colon ACA images:", cancer_colon_aca.shape)

print("Loading non-cancer colon images...")
no_cancer_colon = get_images(colon_n_dir, m, n)
print("Loaded non-cancer colon images:", no_cancer_colon.shape)

# Define the sizes of the training sets
trainSetSize_colon_aca = 4000
trainSetSize_colon_n = 4000

# Colon_aca train and test indices
train_index_colon_aca = np.random.choice(range(5000), trainSetSize_colon_aca, replace=False)
test_index_colon_aca = np.setdiff1d(np.arange(5000), train_index_colon_aca)

# Colon_aca train and test matrices
test_set_colon_aca = np.take(cancer_colon_aca, test_index_colon_aca, axis=0)
train_set_colon_aca = np.take(cancer_colon_aca, train_index_colon_aca, axis=0)

# Colon_n train and test indices
train_index_colon_n = np.random.choice(range(5000), trainSetSize_colon_n, replace=False)
test_index_colon_n = np.setdiff1d(np.arange(5000), train_index_colon_n)

# Colon_n train and test matrices
test_set_colon_n = np.take(no_cancer_colon, test_index_colon_n, axis=0)
train_set_colon_n = np.take(no_cancer_colon, train_index_colon_n, axis=0)

test_sets = [
    (test_set_colon_aca, 'Colon_ACA'),
    (test_set_colon_n, 'Colon_N')
]

# Mean-shifted data for colon
mean_colon_data = np.mean(np.vstack((train_set_colon_aca, train_set_colon_n)), axis=0)  # Get average looking colon
mean_shifted_colon_aca = train_set_colon_aca - mean_colon_data  # Subtracting to highlight important features 
mean_shifted_colon_n = train_set_colon_n - mean_colon_data

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
    def __init__ (self, mean_shifted_colon_aca, mean_shifted_colon_n,
                 m, n, r, p, q, randomized=True):
        self.m = m
        self.n = n
        self.r = r

         # Reshape the mean image data into rows (1, m * n)
        self.colon_aca_mu = mean_colon_data.reshape(1, m * n)  # 1 x 2500 (50 x 50)
        self.colon_n_mu = mean_colon_data.reshape(1, m * n)

        if randomized:
            rsvd_instance = RSVD(r, p)
            _, _, self.Vt_colon_aca = rsvd_instance.rsvd(mean_shifted_colon_aca)
            _, _, self.Vt_colon_n = rsvd_instance.rsvd(mean_shifted_colon_n)
        else:
            _, _, self.Vt_colon_aca = svd(mean_shifted_colon_aca, full_matrices=False)
            _, _, self.Vt_colon_n = svd(mean_shifted_colon_n, full_matrices=False)
        
        # Ensure Vt arrays are numpy arrays
        self.Vt_colon_aca = np.array(self.Vt_colon_aca)
        self.Vt_colon_n = np.array(self.Vt_colon_n)
        
        # Project the mean-shifted datasets onto the low-dimensional subspace
        self.colon_aca_projection = self.low_rank_representation(mean_shifted_colon_aca, self.Vt_colon_aca.T) 
        self.colon_n_projection = self.low_rank_representation(mean_shifted_colon_n, self.Vt_colon_n.T)

    def low_rank_representation(self, f, Vt):
        return f @ Vt
        
    def detect_image(self, g):
        g = g.reshape((1, -1)) #reshaping into a 1, m*n
        g_mean_shifted_colon_aca = g - self.colon_aca_mu
        g_mean_shifted_colon_n = g - self.colon_n_mu

        g_projected_colon_aca = self.low_rank_representation(g_mean_shifted_colon_aca, self.Vt_colon_aca.T) 
        g_projected_colon_n = self.low_rank_representation(g_mean_shifted_colon_n, self.Vt_colon_n.T)

        distances_colon_aca = np.linalg.norm(self.colon_aca_projection - g_projected_colon_aca, axis=1)
        distances_colon_n = np.linalg.norm(self.colon_n_projection - g_projected_colon_n, axis=1)

        min_distance_colon_aca = np.min(distances_colon_aca)
        min_distance_colon_n = np.min(distances_colon_n)

        if min_distance_colon_aca < min_distance_colon_n:
            result_category = 'Colon_ACA'
            min_distance = min_distance_colon_aca
        else:
            result_category = 'Colon_N'
            min_distance = min_distance_colon_n

        return result_category, min_distance

m, n, r, p, q = 50, 50, 500, 5, 5

classifier = ImageClassifier(mean_shifted_colon_aca, mean_shifted_colon_n,
                 m, n, r, p, q, randomized=True)

# Select a random test set and then a random image from that set
selected_set, selected_label = test_sets[np.random.randint(len(test_sets))]
random_test_image_index = np.random.randint(len(selected_set))
random_test_image = selected_set[random_test_image_index]

# Mean-shift the random test image based on its label
if selected_label == 'Colon_ACA':
    mean_shifted_random_test_image = random_test_image - mean_colon_data
    random_test_projection = classifier.low_rank_representation(mean_shifted_random_test_image.reshape(1, -1), classifier.Vt_colon_aca.T)
elif selected_label == 'Colon_N':
    mean_shifted_random_test_image = random_test_image - mean_colon_data
    random_test_projection = classifier.low_rank_representation(mean_shifted_random_test_image.reshape(1, -1), classifier.Vt_colon_n.T)
else:
    raise ValueError("Unexpected label for selected test set.")

result_category, min_distance = classifier.detect_image(random_test_image)
print(f"Detected category: {result_category}, Minimum distance: {min_distance}")

# Generate and plot the projections for all the test images
def plot_projections(classifier, test_sets, random_test_projection, random_test_image_index):
    colors = {
        'Colon_ACA': 'r',
        'Colon_N': 'b'
    }
    markers = {
        'Colon_ACA': 'o',
        'Colon_N': '^'
    }

    plt.figure(figsize=(10, 7))

    all_projections = []
    all_labels = []

    for test_set, label in test_sets:
        mean_shifted_test_set = test_set - mean_colon_data
        if label == 'Colon_ACA':
            projections = classifier.low_rank_representation(mean_shifted_test_set, classifier.Vt_colon_aca.T)
        elif label == 'Colon_N':
            projections = classifier.low_rank_representation(mean_shifted_test_set, classifier.Vt_colon_n.T)
        else:
            raise ValueError("Unexpected label for test set.")
        
        all_projections.append(projections)
        all_labels.append([label] * len(projections))
        
        indices = np.arange(len(projections))
        plt.scatter(projections[indices, 0], projections[indices, 1], 
                    c=colors[label], marker=markers[label], label=label, alpha=0.5)

    # Convert to numpy arrays for easier manipulation
    all_projections = np.vstack(all_projections)
    all_labels = np.hstack(all_labels)

    # Highlight the random test image
    plt.scatter(random_test_projection[:, 0], random_test_projection[:, 1], 
                c='k', marker='*', s=200, label='Random Test Image')

    plt.xlabel('Projection Dimension 1')
    plt.ylabel('Projection Dimension 2')
    plt.title('Projections of Test Images')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_projections(classifier, test_sets, random_test_projection, random_test_image_index)
