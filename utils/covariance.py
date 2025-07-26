import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('d_user1.pkl', 'rb') as f:
    d = pickle.load(f)

d = np.array(d.cpu().numpy())

# Covariance between samples (default)
cov_samples = np.cov(d)
print("Covariance between samples (shape {}):".format(cov_samples.shape))
print(cov_samples)

# Covariance between attributes (columns)
cov_attr = np.cov(d, rowvar=False)
print("Covariance between attributes (shape {}):".format(cov_attr.shape))
print(cov_attr)

plt.figure(figsize=(8, 6))
plt.imshow(cov_attr, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Covariance')
plt.title('Covariance Matrix of Attributes (columns of d)')
plt.xlabel('Attribute')
plt.ylabel('Attribute')
plt.tight_layout()
plt.savefig('covariance_matrix_d_attributes.png', dpi=200)
plt.show()