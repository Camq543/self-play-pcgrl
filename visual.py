import matplotlib.pyplot as plt
import re

d = open("logs\zelda_turtle_map_restricted_log.txt", "r") #specify log file here

s = d.read()
x = re.findall('\{(.*?)\}',s)
y = re.findall("'reward': \[(.*?\])",s) #specify feature name here
num = str(y)[1:-1]

data_flat = re.findall(r'\d+\.\d+',num) #extract raw floats numbers from feature for agent1 and agent2

y_agent1_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 == 0] 
print(y_agent1_val)
print("")
y_agent2_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 != 0] 
print(y_agent2_val)
print("")

number_of_obs = len(y_agent1_val)
print("Number of updates: " + str(number_of_obs))


x_val_og = range(number_of_obs)
x_val = [element * 50 for element in x_val_og]  #update interval is 50

plt.plot(x_val, y_agent1_val)
plt.plot(x_val, y_agent2_val)
plt.xlabel('Update Interval')
plt.ylabel('Value')

plt.show()




