import numpy as np
from scipy.stats import mode    # for mode, aka majority voting
from sklearn.preprocessing import normalize  # for vector normalization
import pandas as pd  # For csv handling

# ===============================================
# Intro to Numpy (1)
# ===============================================


# init a numpy array of shape (100,1) sampling from uniform distro within 0 and 1
xx = np.random.uniform(low=0.0, high=1.0, size=100)

# mean of the elements in xx
xx.mean()

# sum of the elements in xx
xx.sum()

# mode of the elements in xx (majority votes)
mode, count = stats.mode(xx, axis=0)
print('mode of xx is', mode[0], 'with occurance' count[0])

# Print the first 10 and last 10 array of the array
print(xx[:10])
print(xx[-10:])

# Flip this 1D array, aka flip the axis=0 row of array
np.flip(xx, axis=0)

# Sort the 1D array in descending order, aka sort along the first axis
np.sort(xx, axis=0)

# ===============================================
# Intro to Numpy (2)
# ===============================================

# Init a 5*5 matrix
M = np.random.uniform(low=1, high=10, size=(5, 10))

M.reshape(2, 5, 5)

# Calculate the determinant
np.linalg.det(M)

# Check if inverse exists
np.linalg.matrix_rank(M) == M.shape[0]
np.linalg.inv(M)

# Sort M along row (first axis), along the column (second axis) and then on falttend
np.sort(M, axis=0)
np.sort(M, axis=0)
np.sort(M, axis=None)

# Say each column is a single feature, and rows represents samples. Nomalize the feature values.
normalize(M, axis=0)
# The normalize function in sklearn.preprocessing is used for normalization: axis=1 for indivisual sampela and axis=0 for entire features


# ===============================================
# Intro to Pandas
# ==============================================

dataframe = pd.read_csv('./exercises/student_marks.csv', delimiter=',')

# Below is a trick for finding current working dir
# from pathlib import Path
# >>> cwd = Path.cwd()
# >>> cwd
# WindowsPath('D:/Github/ai4good')

# Check the headers of the dataframe (in list), same result of list(dataframe) but way faster
dataframe.columns.values.tolist()

# How to mannualy skip the first row:
dataframe = pd.read_csv('./exercises/student_marks.csv',
                        delimiter=',', skiprows=1)

# How to mannulay modify header names
headers = dataframe.columns.values.tolist()
headers[0] = 'student'
dataframe.columns = headers

# To view a small sample of the dataframe
dataframe.head()
dataframe.tail()

# To modify a specific cell in the dataframe
# CF: https://stackoverflow.com/questions/13842088/set-value-for-particular-cell-in-pandas-dataframe-using-index

# Accessing cells
# CF: https://stackoverflow.com/questions/31593201/how-are-iloc-ix-and-loc-different
dataframe.iloc[0][2]
dataframe.iloc[0][:]

# Calculate percentage of mark for each student
grades = dataframe.iloc[:, -8:]
total_grades = grades.sum(axis=1)
avg_grades = total_grades.div(8)
# Add the avg_grades to the original dataframe
dataframe['percentage'] = avg_grades

# Bin the percentage of each student into grades:
# CF: https://jeffdelaney.me/blog/useful-snippets-in-pandas/
bins = [0, 60, 70, 75, 85, 90, 100]
names = ['F', 'C', 'B-', 'B', 'A-', 'A']
df['grade'] = pd.cut(df.percentage, bins, labels=names)


# Top 3 performers overall
# Drop a column with label XXX:
# df.drop(['grade'], axis=1)
# Sorting according to a column
df.sort_values('grade', axis=0, ascending=False)

top3 = df.iloc[:3, 0]

# Logic conditional
# CF: https://stackoverflow.com/questions/41534428/how-do-i-perform-a-math-operation-on-a-python-pandas-dataframe-column-but-only


# Modify the value of a specific cell:
dataframe.at[0, 'Gender'] = 'F'
dataframe.at[0, 'Gender'] = 'M'

# Pandas dataframe plotting
# CF: https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
df.percentage.plot(kind='kde')

# groupby and then plot
df.groupby('Gender').percentage.plot(kind='kde'
                                     )
