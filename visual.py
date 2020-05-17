import matplotlib.pyplot as plt
import re
import os
from scipy.ndimage.filters import gaussian_filter1d


game = 'binary'
representation = 'narrow'

map_restricted = False 
negative_switch = False
self_play = True

sp = 'self_play_' if self_play else ''
mr = 'map_restricted_' if map_restricted else ''
ns = 'negative_switch_' if negative_switch else ''

filename = "logs/{}{}_{}_{}{}log.txt".format(sp,game, representation, mr, ns)

if game == 'zelda':
	####### REWARD PLOT #######
	

	d = open(filename, "r") #CHANGE THIS

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

	ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
	ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

	#plt.figure(figsize=(12,9))
	plt.plot(x_val, ysmoothed_1, label = "Agent 1")
	plt.plot(x_val, ysmoothed_2, label = "Agent 2")
	plt.title("{} {} {}{}{}- Reward Values".format(game, 
												representation, 
												'Self Play ' if self_play else '', 
												'Map Restricted ' if map_restricted else '',
												'Negative Switch ' if negative_switch else ''))
	plt.legend(loc="lower right")
	plt.xlabel('Update Interval')
	plt.ylabel('Reward') #CHANGE THIS

	strFile = "./figs/{}{}_{}_{}{}reward.pdf".format(sp,game, representation, mr, ns) #CHANGE THIS
	if os.path.isfile(strFile):
	   os.remove(strFile)
	plt.savefig(strFile)
	#plt.show()
	plt.clf()
	d.close()

	####### KEY PLOT #######
	h = open(filename, "r")  #CHANGE THIS

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

	ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
	ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

	#plt.figure(figsize=(12,9))
	plt.plot(x_val, ysmoothed_1, label = "Agent 1")
	plt.plot(x_val, ysmoothed_2, label = "Agent 2")

	plt.title("{} {} {}{}{}- Reward Values".format(game, 
												representation, 
												'Self Play ' if self_play else '', 
												'Map Restricted ' if map_restricted else '',
												'Negative Switch ' if negative_switch else ''))
	plt.legend(loc="upper right")
	plt.xlabel('Update Interval')
	plt.ylabel('Key') #CHANGE THIS

	strFile = "./figs/{}{}_{}_{}{}key.pdf".format(sp,game, representation, mr, ns) #CHANGE THIS
	if os.path.isfile(strFile):
	   os.remove(strFile)
	plt.savefig(strFile)
	#plt.show()
	plt.clf()
	h.close()

	####### DOOR PLOT #######
	u = open(filename, "r") #CHANGE THIS

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

	ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
	ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

	#plt.figure(figsize=(12,9))
	plt.plot(x_val, ysmoothed_1, label = "Agent 1")
	plt.plot(x_val, ysmoothed_2, label = "Agent 2")

	plt.title("{} {} {}{}{}- Reward Values".format(game, 
												representation, 
												'Self Play ' if self_play else '', 
												'Map Restricted ' if map_restricted else '',
												'Negative Switch ' if negative_switch else ''))

	plt.legend(loc="upper right")
	plt.xlabel('Update Interval')
	plt.ylabel('Door') #CHANGE THIS

	strFile = "./figs/{}{}_{}_{}{}door.pdf".format(sp,game, representation, mr, ns) #CHANGE THIS
	if os.path.isfile(strFile):
	   os.remove(strFile)
	plt.savefig(strFile)
	#plt.show()
	plt.clf()
	u.close()

else:
	####### REWARD PLOT #######

	d = open(filename, "r") #CHANGE THIS

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

	ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
	ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

	#plt.figure(figsize=(12,9))
	plt.plot(x_val, ysmoothed_1, label = "Agent 1")
	plt.plot(x_val, ysmoothed_2, label = "Agent 2")
	plt.title("{} {} {}{}{}- Reward Values".format(game, 
												representation, 
												'Self Play ' if self_play else '', 
												'Map Restricted ' if map_restricted else '',
												'Negative Switch ' if negative_switch else '')) 
	plt.legend(loc="lower right")
	plt.xlabel('Update Interval')
	plt.ylabel('Reward') #CHANGE THIS

	strFile = "./figs/{}{}_{}_{}{}reward.pdf".format(sp,game, representation, mr, ns) #CHANGE THIS
	if os.path.isfile(strFile):
	   os.remove(strFile)
	plt.savefig(strFile)
	#plt.show()
	plt.clf()
	d.close()

	####### PATH CHANGE PLOT #######
	h = open(filename, "r")  #CHANGE THIS

	q = h.read()
	print(q)
	x = re.findall('\{(.*?)\}',q)
	y = re.findall("'path-change': \[(.*?\])",q) 
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

	ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
	ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

	#plt.figure(figsize=(12,9))
	plt.plot(x_val, ysmoothed_1, label = "Agent 1")
	plt.plot(x_val, ysmoothed_2, label = "Agent 2")

	plt.title("{} {} {}{}{}- Reward Values".format(game, 
												representation, 
												'Self Play ' if self_play else '', 
												'Map Restricted ' if map_restricted else '',
												'Negative Switch ' if negative_switch else ''))
	plt.legend(loc="upper right")
	plt.xlabel('Update Interval')
	plt.ylabel('Path length change') #CHANGE THIS

	strFile = "./figs/{}{}_{}_{}{}path_change.pdf".format(sp,game, representation, mr, ns) #CHANGE THIS
	if os.path.isfile(strFile):
	   os.remove(strFile)
	plt.savefig(strFile)
	#plt.show()
	plt.clf()
	h.close()

	####### PATH IMPROVEMENT PLOT #######
	u = open(filename, "r") #CHANGE THIS

	w = u.read()
	x = re.findall('\{(.*?)\}',w)
	y = re.findall("'path-imp': \[(.*?\])",w) #specify feature name here
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

	ysmoothed_1 = gaussian_filter1d(y_agent1_val, sigma=2)
	ysmoothed_2 = gaussian_filter1d(y_agent2_val, sigma=2)

	#plt.figure(figsize=(12,9))
	plt.plot(x_val, ysmoothed_1, label = "Agent 1")
	plt.plot(x_val, ysmoothed_2, label = "Agent 2")

	plt.title("{} {} {}{}{}- Reward Values".format(game, 
												representation, 
												'Self Play ' if self_play else '', 
												'Map Restricted ' if map_restricted else '',
												'Negative Switch ' if negative_switch else ''))

	plt.legend(loc="upper right")
	plt.xlabel('Update Interval')
	plt.ylabel('Path length improvement') #CHANGE THIS

	strFile = "./figs/{}{}_{}_{}{}path_imp.pdf".format(sp,game, representation, mr, ns) #CHANGE THIS
	if os.path.isfile(strFile):
	   os.remove(strFile)
	plt.savefig(strFile)
	#plt.show()
	plt.clf()
	u.close()
