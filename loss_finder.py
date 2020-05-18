import matplotlib.pyplot as plt
import re
import os
import json
from scipy.ndimage.filters import gaussian_filter1d



game = 'zelda'
representation = 'narrow'

map_restricted = False 
negative_switch = True
self_play = True

sp = 'self_play_' if self_play else ''
mr = 'map_restricted_' if map_restricted else ''
ns = 'negative_switch_' if negative_switch else ''

filename = "logs/{}{}_{}_{}{}log.txt".format(sp,game, representation, mr, ns)
print(filename)

d = open(filename, "r") #CHANGE THIS

array_list = []

for line in d:
	if '{' in line:
		continue
	else:
		if line[0] == '[': 
			tempstr = line.strip()

		while tempstr[-3:] != '])]':
			line = d.readline()
			tempstr += line.strip()
			# print(tempstr)

		tempstr = tempstr.replace('array','').replace('(','').replace(')','')
		array_list.append(json.loads(tempstr))

loss_list = []

for train_info in array_list:
	temp = []
	for vals in train_info:
		loss = -(vals[0] - .5 * vals[1] + .01 * vals[2])
		temp.append(loss)
	loss_list.append(temp)


print(loss_list)
print(len(loss_list))
print(type(loss_list[0][0]))
# s = d.read()
# x = re.findall('\{(.*?)\}',s)
# y = re.findall("\[array\(\[.*?\]\)\]",s) #specify feature name here

# print(y)
# print(len(y))
