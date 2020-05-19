import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

import re
import os
from scipy.ndimage.filters import gaussian_filter1d

####### REWARD PLOT #######
d = open("logs/binary_narrow_log.txt", "r") #CHANGE THIS

s = d.read()
x = re.findall('\{(.*?)\}',s)
y = re.findall("'reward': \[(-?.*?\])",s) #specify feature name here
num = str(y)[1:-1]

data_flat = re.findall(r'-?\d+\.\d+',num) #extract raw floats numbers from feature for agent1 and agent2

y_agent1_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 == 0] 
for i in range(0, len(y_agent1_val)): 
    y_agent1_val[i] = float(y_agent1_val[i]) 
print(y_agent1_val)
print("")

y_agent2_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 != 0] 
for i in range(0, len(y_agent2_val)): 
    y_agent2_val[i] = float(y_agent2_val[i]) 
print(y_agent2_val)
print("")

number_of_obs = len(y_agent1_val)
print("Number of updates: " + str(number_of_obs))

x_val_og = range(number_of_obs)
x_val = [element * 50 for element in x_val_og]  #update interval is 50

ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

plt.figure(figsize=(12,9))

x_valc = x_val[0:200]
ysmoothed_1c = ysmoothed_1[0:200]
ysmoothed_2c = ysmoothed_2[0:200]

plt.plot(x_valc, ysmoothed_1c, label = "Narrow (Agent 1)", color = "#5899DA")
plt.plot(x_valc, ysmoothed_2c, label = "Narrow (Agent 2)", color = "#5899DA", linestyle='--')
#plt.title("Self Play Binary Path Change Values") #CHANGE THIS
#plt.legend(loc="lower right")
plt.xlabel('Update Interval')
plt.ylabel('Path Improvement') #CHANGE THIS

#strFile = "./figs/zelda_turtle_map_restricted_reward.png" #CHANGE THIS
#if os.path.isfile(strFile):
#   os.remove(strFile)
#plt.savefig(strFile)
#plt.show()
#plt.clf()
d.close()

####### KEY PLOT #######
h = open("logs/binary_narrow_log_single_agent.txt", "r") #CHANGE THIS

q = h.read()
print(q)
x = re.findall('\{(.*?)\}',q)
y = re.findall("'reward': \[(-?.*?\])",q) 
num = str(y)[1:-1]

data_flat = re.findall(r'-?\d+\.\d+',num) #extract raw floats numbers from feature for agent1 and agent2

y_agent1_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 == 0] 
for i in range(0, len(y_agent1_val)): 
    y_agent1_val[i] = float(y_agent1_val[i]) 
print(y_agent1_val)
print("")

y_agent2_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 != 0] 
for i in range(0, len(y_agent2_val)): 
    y_agent2_val[i] = float(y_agent2_val[i]) 
print(y_agent2_val)
print("")

number_of_obs = len(y_agent1_val)
print("Number of updates: " + str(number_of_obs))

x_val_og = range(number_of_obs)
x_val = [element * 50 for element in x_val_og]  #update interval is 50

ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

x_valc = x_val[0:200]
ysmoothed_1c = ysmoothed_1[0:200]
ysmoothed_2c = ysmoothed_2[0:200]

#plt.figure(figsize=(12,9))
plt.plot(x_valc, ysmoothed_1c, label = "Narrow Single Agent (Agent 1)", color = '#E8743B')
#plt.plot(x_valc, ysmoothed_2c, color = '#E8743B', linestyle='--')

#plt.title("Zelda Turtle Map Restricted - Key Values") #CHANGE THIS
#plt.legend(loc="upper right")
#plt.xlabel('Update Interval')
#plt.ylabel('Key') #CHANGE THIS

#strFile = "./figs/zelda_turtle_map_restricted_key.png" #CHANGE THIS
#if os.path.isfile(strFile):
#   os.remove(strFile)
#plt.savefig(strFile)
#plt.show()
#plt.clf()
h.close()

####### DOOR PLOT #######
u = open("logs/binary_narrow_map_restricted_log.txt", "r") #CHANGE THIS

w = u.read()
x = re.findall('\{(.*?)\}',w)
y = re.findall("'reward': \[(-?.*?\])",w) #specify feature name here
num = str(y)[1:-1]

data_flat = re.findall(r'-?\d+\.\d+',num) #extract raw floats numbers from feature for agent1 and agent2

y_agent1_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 == 0] 
for i in range(0, len(y_agent1_val)): 
    y_agent1_val[i] = float(y_agent1_val[i]) 
print(y_agent1_val)
print("")

y_agent2_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 != 0] 
for i in range(0, len(y_agent2_val)): 
    y_agent2_val[i] = float(y_agent2_val[i]) 
print(y_agent2_val)
print("")

number_of_obs = len(y_agent1_val)
print("Number of updates: " + str(number_of_obs))

x_val_og = range(number_of_obs)
x_val = [element * 50 for element in x_val_og]  #update interval is 50

ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

x_vald = x_val[0:200]
ysmoothed_1d = ysmoothed_1[0:200]
ysmoothed_2d = ysmoothed_2[0:200]


#plt.figure(figsize=(12,9))
plt.plot(x_vald, ysmoothed_1d, label = "Narrow Map Restricted (Agent 1)", color = '#19A979')
plt.plot(x_vald, ysmoothed_2d, label = "Narrow Map Restricted (Agent 2)", color = '#19A979', linestyle='--')

#plt.title("Zelda Turtle Map Restricted - Door Values") #CHANGE THIS
#plt.legend(loc="upper right")
#plt.xlabel('Update Interval')
#plt.ylabel('Door') #CHANGE THIS

#strFile = "./figs/zelda_turtle_map_restricted_door.png" #CHANGE THIS
#if os.path.isfile(strFile):
#  os.remove(strFile)
#plt.savefig(strFile)
#plt.show()
#plt.clf()
u.close()



####### DOOR PLOT #######
u = open("logs/binary_narrow_negative_switch_log.txt", "r") #CHANGE THIS

w = u.read()
x = re.findall('\{(.*?)\}',w)
y = re.findall("'reward': \[(-?.*?\])",w) #specify feature name here
num = str(y)[1:-1]

data_flat = re.findall(r'-?\d+\.\d+',num) #extract raw floats numbers from feature for agent1 and agent2

y_agent1_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 == 0] 
for i in range(0, len(y_agent1_val)): 
    y_agent1_val[i] = float(y_agent1_val[i]) 
print(y_agent1_val)
print("")

y_agent2_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 != 0] 
for i in range(0, len(y_agent2_val)): 
    y_agent2_val[i] = float(y_agent2_val[i]) 
print(y_agent2_val)
print("")

number_of_obs = len(y_agent1_val)
print("Number of updates: " + str(number_of_obs))

x_val_og = range(number_of_obs)
x_val = [element * 50 for element in x_val_og]  #update interval is 50

ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

x_vald = x_val[0:200]
ysmoothed_1d = ysmoothed_1[0:200]
ysmoothed_2d = ysmoothed_2[0:200]


#plt.figure(figsize=(12,9))
plt.plot(x_vald, ysmoothed_1d, label = "Narrow Negative Switch (Agent 1)", color = '#ED4A7B')
plt.plot(x_vald, ysmoothed_2d, label = "Narrow Negative Switch (Agent 2)", color = '#ED4A7B', linestyle='--')

#plt.title("Zelda Turtle Map Restricted - Door Values") #CHANGE THIS
#plt.legend(loc="upper right")
#plt.xlabel('Update Interval')
#plt.ylabel('Door') #CHANGE THIS

#strFile = "./figs/zelda_turtle_map_restricted_door.png" #CHANGE THIS
#if os.path.isfile(strFile):
#  os.remove(strFile)
#plt.savefig(strFile)
#plt.show()
#plt.clf()
u.close()


####### DOOR PLOT #######
u = open("logs/binary_turtle_map_restricted_log.txt", "r") #CHANGE THIS

w = u.read()
x = re.findall('\{(.*?)\}',w)
y = re.findall("'reward': \[(-?.*?\])",w) #specify feature name here
num = str(y)[1:-1]

data_flat = re.findall(r'-?\d+\.\d+',num) #extract raw floats numbers from feature for agent1 and agent2

y_agent1_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 == 0] 
for i in range(0, len(y_agent1_val)): 
    y_agent1_val[i] = float(y_agent1_val[i]) 
print(y_agent1_val)
print("")

y_agent2_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 != 0] 
for i in range(0, len(y_agent2_val)): 
    y_agent2_val[i] = float(y_agent2_val[i]) 
print(y_agent2_val)
print("")

number_of_obs = len(y_agent1_val)
print("Number of updates: " + str(number_of_obs))

x_val_og = range(number_of_obs)
x_val = [element * 50 for element in x_val_og]  #update interval is 50

ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

x_vald = x_val[0:200]
ysmoothed_1d = ysmoothed_1[0:200]
ysmoothed_2d = ysmoothed_2[0:200]


#plt.figure(figsize=(12,9))
plt.plot(x_vald, ysmoothed_1d, label = "Turtle Map Restricted (Agent 1)", color = '#945ECF')
plt.plot(x_vald, ysmoothed_2d, label = "Turtle Map Restricted (Agent 2)", color = '#945ECF', linestyle='--')

#plt.title("Zelda Turtle Map Restricted - Door Values") #CHANGE THIS
#plt.legend(loc="upper right")
#plt.xlabel('Update Interval')
#plt.ylabel('Door') #CHANGE THIS

#strFile = "./figs/zelda_turtle_map_restricted_door.png" #CHANGE THIS
#if os.path.isfile(strFile):
#  os.remove(strFile)
#plt.savefig(strFile)
#plt.show()
#plt.clf()
u.close()






####### DOOR PLOT #######
j = open("logs/binary_turtle_negative_switch_log.txt", "r") #CHANGE THIS

w = j.read()
x = re.findall('\{(.*?)\}',w)
y = re.findall("'reward': \[(-?.*?\])",w) #specify feature name here
num = str(y)[1:-1]

data_flat = re.findall(r'-?\d+\.\d+',num) #extract raw floats numbers from feature for agent1 and agent2

y_agent1_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 == 0] 
for i in range(0, len(y_agent1_val)): 
    y_agent1_val[i] = float(y_agent1_val[i]) 
print(y_agent1_val)
print("")

y_agent2_val = [data_flat[i] for i in range(len(data_flat)) if i % 2 != 0] 
for i in range(0, len(y_agent2_val)): 
    y_agent2_val[i] = float(y_agent2_val[i]) 
print(y_agent2_val)
print("")

number_of_obs = len(y_agent1_val)
print("Number of updates: " + str(number_of_obs))

x_val_og = range(number_of_obs)
x_val = [element * 50 for element in x_val_og]  #update interval is 50

ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

x_valc = x_val[0:200]
ysmoothed_1c = ysmoothed_1[0:200]
ysmoothed_2c = ysmoothed_2[0:200]


#plt.figure(figsize=(12,9))
plt.plot(x_valc, ysmoothed_1c, label = "Turtle Negative Switch (Agent 1)", color = '#13A4B4')
plt.plot(x_valc, ysmoothed_2c, label = "Turtle Negative Switch (Agent 2)", color = '#13A4B4', linestyle='--')

plt.title("Base - Binary") #CHANGE THIS
plt.legend(loc="lower left")
plt.xlabel('Update Interval')
plt.ylabel('Reward Values') #CHANGE THIS

strFile = "./figs/base_binary_reward.png" #CHANGE THIS
if os.path.isfile(strFile):
  os.remove(strFile)
plt.savefig(strFile)
plt.show()
plt.clf()
u.close()

