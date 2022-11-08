import pickle
import os

print(os.listdir(".")) # show all files in the cwd

with open('x_test_object.pkl', 'rb') as in_file:
    x_test = pickle.load(in_file)

print(type(x_test))
print(x_test)