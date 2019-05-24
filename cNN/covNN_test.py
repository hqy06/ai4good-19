import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3d plot
from sklearn.decomposition import PCA                 # PCA
import pandas as pd
import numpy as np
import covNN
from torchvision import transforms

print('********* LOADING CVS FILE\n')
train_df = pd.read_csv('../datasets/digits_train.csv')


print("********* Show_images TEST\n")
images = np.arange(48).reshape(3, 4, 4)
size = (4, 4)
labels = [1, 2, 3]
covNN.show_images(images, size, labels, cols=3)

print('********* DATA AUGMENTATION TEST')
rotate = transforms.RandomRotation(degrees=45)
shift = transforms.RandomAffine(
    degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2))
composed = transforms.Compose([rotate, shift])

rand_indices = np.random.randint(train_df.shape[0], size=3)
samples = train_df.iloc[rand_indices,
                        1:].values.reshape(-1, 28, 28).astype(np.uint8)

t_samples, t_names = [], []
for s in samples:
    img = transforms.ToPILImage()(s)
    for t in [rotate, shift, composed]:
        t_samples.append(np.array(t(img)))
        t_names.append(type(t).__name__)

covNN.show_images(t_samples, (28, 28), t_names, cols=3)


print('********* visualization: PCA')
X_train = train_df.iloc[:, 1:].values
X_train.shape
pca = PCA(n_components=3)

pca_result = pca.fit_transform(X_train)

pca_result.shape

pca_1 = pca_result[:, 0]
pca_2 = pca_result[:, 1]
pca_3 = pca_result[:, 2]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(pca_1, pca_2, pca_3, c=pca_3, cmap='tab10', alpha=0.5)
plt.show()
