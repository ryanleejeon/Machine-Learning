import numpy as np

# For Plotting:
import matplotlib.pyplot as plt
import seaborn as sea


print("Part 1: Supervised learning")
'''
Supervised Learning: 
We can forecast the attendance problem by training a supervised learning algorithm with some labeled examples. Let's open up the data file and see what that looks like.
'''

X,Y = np.loadtxt("Data/Fitness_Class_Data.txt", skiprows = 1, unpack = True)

# Print out first few values
X[0:5]
Y[0:5]

sea.set()
plt.axis([0,30, 0,30])
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel("Registrations", fontsize = 14)
plt.ylabel("Participants", fontsize = 14)

plt.plot(X,Y, "bo")
plt.show()

# Now you can see the plot! 

# Based on what I see, it is clear that the more registrations, the more people.
# But also, participants are generally lower than registrations.


# Let's ignore that these are a measly number of points and try to build a supervised machine learning model from this
