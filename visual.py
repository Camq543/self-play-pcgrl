import matplotlib.pyplot as plt
import re
import os

####### REWARD PLOT #######
d = open("logs\zelda_turtle_map_restricted_log.txt", "r") #CHANGE THIS

s = d.read()
x = re.findall('\{(.*?)\}',s)
y = re.findall("'reward': \[(.*?\])",s) #specify feature name here
num = str(y)[1:-1]

data_flat = re.findall(r'\d+\.\d+',num) #extract raw floats numbers from feature for agent1 and agent2

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

#plt.figure(figsize=(12,9))
plt.plot(x_val, y_agent1_val, label = "Agent 1")
plt.plot(x_val, y_agent2_val, label = "Agent 2")
plt.title("Zelda Turtle Map Restricted - Reward Values") #CHANGE THIS
plt.legend(loc="lower right")
plt.xlabel('Update Interval')
plt.ylabel('Reward') #CHANGE THIS

strFile = "./figs/zelda_turtle_map_restricted_reward.png" #CHANGE THIS
if os.path.isfile(strFile):
   os.remove(strFile)
plt.savefig(strFile)
#plt.show()
plt.clf()
d.close()

####### KEY PLOT #######
h = open("logs\zelda_turtle_map_restricted_log.txt", "r") #CHANGE THIS

q = h.read()
print(q)
x = re.findall('\{(.*?)\}',q)
y = re.findall("'key': \[(.*?\])",q) 
num = str(y)[1:-1]

data_flat = re.findall(r'\d+\.\d+',num) #extract raw floats numbers from feature for agent1 and agent2

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

#plt.figure(figsize=(12,9))
plt.plot(x_val, y_agent1_val, label = "Agent 1")
plt.plot(x_val, y_agent2_val, label = "Agent 2")
plt.title("Zelda Turtle Map Restricted - Key Values") #CHANGE THIS
plt.legend(loc="upper right")
plt.xlabel('Update Interval')
plt.ylabel('Key') #CHANGE THIS

strFile = "./figs/zelda_turtle_map_restricted_key.png" #CHANGE THIS
if os.path.isfile(strFile):
   os.remove(strFile)
plt.savefig(strFile)
#plt.show()
plt.clf()
h.close()

####### DOOR PLOT #######
u = open("logs\zelda_turtle_map_restricted_log.txt", "r") #CHANGE THIS

w = u.read()
x = re.findall('\{(.*?)\}',w)
y = re.findall("'door': \[(.*?\])",w) #specify feature name here
num = str(y)[1:-1]

data_flat = re.findall(r'\d+\.\d+',num) #extract raw floats numbers from feature for agent1 and agent2

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

#plt.figure(figsize=(12,9))
plt.plot(x_val, y_agent1_val, label = "Agent 1")
plt.plot(x_val, y_agent2_val, label = "Agent 2")
plt.title("Zelda Turtle Map Restricted - Door Values") #CHANGE THIS
plt.legend(loc="upper right")
plt.xlabel('Update Interval')
plt.ylabel('Door') #CHANGE THIS

strFile = "./figs/zelda_turtle_map_restricted_door.png" #CHANGE THIS
if os.path.isfile(strFile):
   os.remove(strFile)
plt.savefig(strFile)
#plt.show()
plt.clf()
u.close()

