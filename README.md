## Optimizing face recognition with image compression (PCA)

In this project, principal component analysis (PCA) is applied to reduce the dimension of a dataset of images in order to optimize a subsequent classification task. Supervised ML task: Given a a picture of a face of a famous person, tell to whom the face belongs (7 classes). For this task, we will use support vector classification support vector classification (SVC)

**Procedure** 
* The dataset: Feth_lfw_people image dataset from sklearn including  7 classes. 
	*Each class is a famous person: Ariel Sharon, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Hugo Chavez, Tony Blair
* Data structure:  
- `faces.images` images as matrices of **50 x 37 pixels** (plottable) 
- `faces.data` flattened version of size **1850 x 1** *(50 x 37=1850)* 
- `faces.target` number index representing a class among 7

* Optimal number of principal components: The dataset contains 1288 images and 1850 features (50 × 37 pixels), a bad ratio for machine learning. We perform a grid search in order to look for the optimal number of components (n = 200).

* Baseline model accuracy = ~52%
* Scaling:  StandardScaler from sklearn
* Balancing: Representation of classes is unbalanced, so we train a second pipeline that takes into account the class imbalance with the argument class_weight of SVC

* Final fine-tuning: we gridsearch three hyperparameters of the svc architecture (kernel, gamma and C) to improve the model's performance.  

