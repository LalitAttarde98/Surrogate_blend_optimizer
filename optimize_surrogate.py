import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import rdkit
from scipy.optimize import minimize
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def find_group(s):
    double_bonds = len(re.findall(r'C=C',s))
    #------------------------------------------------------
    iter = re.finditer(r"C=C", s)
    indices = set()
    for m in iter:
        if m.start(0) == 0:
            indices.add(m.end(0))
        if m.end(0) == len(s):
            indices.add(m.start(0) - 1)
        if m.start(0) != 0 and m.end(0) != len(s):
            indices.add(m.start(0)-1)
            indices.add(m.end(0))
    try:
        indices.remove(s.find("C(=O)OC",0,len(s)))
    except Exception:
        print("_________")
    allylic = len(indices)
    #------------------------------------------------------
    if 0 in indices or len(s)-1 in indices:
        CH3 = len(re.findall(r'^CC|CC$',s,overlapped=True)) - 1
    else:
        CH3 = len(re.findall(r'^CC|CC$',s,overlapped=True))
    #--------------------------------------------------------
    if CH3 == 2:
        new = s[1:-1]
    elif CH3 == 1:
        if s[:2] == 'CC':
            new = s[1:]
        else:
            new = s[:-1]
    else:
        new = s
    new = new.replace("C(=O)OC","")
    new = new.replace("C=C","")
    if 0 in indices or len(s)-1 in indices:
        CH2 = len(new) - allylic + len(re.findall(r'^CC=C|C=CC$',s)) - 1
    else:
        CH2 = len(new) - allylic + len(re.findall(r'^CC=C|C=CC$',s))
    #--------------------------------------------------------------------
    max_length = 0
    current_length = 0
    string = s.replace("C(=O)OC","")
    string = s.replace("C=C","=")
    for char in string:
        if char == 'C':
            current_length += 1
        else:
            max_length = max(max_length, current_length)
            current_length = 0
    max_length = max(max_length, current_length)
    #--------------------------------------------------------------
    COOCH3 = s.count('C(=O)OC',0,len(s)) + s.count('COC(=O)',0,len(s))

    C = len(re.findall(r'C',s))
    O = len(re.findall(r'O',s))
    my_mol = Chem.MolFromSmiles(s)
    H_string = rdMolDescriptors.CalcMolFormula(my_mol)
    try:
        H = int(re.findall("(?<=H)(.*?)(?=O)",H_string)[0])
    except Exception:
        H = int(re.findall("[^H]*",H_string)[2])
    return [CH3,CH2,allylic,double_bonds,COOCH3,C,H,O]
    
def find_target(Target_fuel):
	if isinstance(Target_fuel, str):
		y = np.append(find_group(Target_fuel), np.asarray(Properties[Target_fuel]))
	elif isinstance(Target_fuel, tuple):
		temp = []
		temp.append([find_group(smile) for smile in Biodiesel_comp])
		y = np.append(np.asarray(np.asarray(Target_fuel) @ np.asarray(temp)), np.asarray(Properties[Target_fuel]))
	elif isinstance(Target_fuel, dict):
		prop = []
		for smile in Target_fuel:
			if isinstance(smile, str):
				temp = np.append(find_group(smile), np.asarray(Properties[smile]))
			else:
				temp = np.append([np.asarray(smile) @ np.asarray([find_group(i) for i in Biodiesel_comp])] , Properties[smile])
			MW = (temp[5]*12) + temp[6] + (temp[7]*16)
			temp = np.append(temp,MW/Properties[smile][2])
			temp = np.append(temp,MW)
			temp = np.append(temp,MW/Properties[smile][2])
			prop.append(temp)
		prop = np.asarray(prop).T
		values = np.array(list(Target_fuel.values()))
		y = values @ prop[:8].T
		y = np.append(y,np.sum(prop[8:11] * prop[11:14] * values, axis = 1) / np.sum(prop[11:14] * values, axis = 1))
	return y.flatten()[:8]

def optmize(A, y):
	def objective_function(x):
	    loss =  (A @ x) - y
	    return np.nansum(loss ** 2)
  
	x0 = [1.0] * len(A[0])
         
	bounds = ((0.001, 10.0) for i in range(len(x0)))  

	result = minimize(objective_function, x0, bounds=bounds)
	
	print("Optimal moles: ",result.x," & mole fractions:",result.x/sum(result.x))
	print("Minimum loss:", result.fun)
	return result.x

def plot_analysis(x,A,y):
    ###############################
    plt.figure()
    plt.title("Comparison of predicted and actual targets: (actual, predicted)")
    plt.plot(A @ x)
    plt.plot(y)
    plt.legend(["Predicted", "Actual"],loc='upper left')
    plt.xlabel("Types of functional groups")
    plt.xticks(np.arange(8), ["CH3", "CH2", "allylic", "double_bonds", "COOCH3", "C", "H", "O"], fontsize=8)
    plt.ylabel("units")
    for i_x, i_y, sol in zip(range(0,len(y)+1), y,  A @ x):
        plt.text(i_x, i_y, '({},{})'.format(round(i_y,2),round(sol,2)))
    plt.show()
    ############################

######################################################## Data Library ##############################################################

R = 8.314 

methyl_butanoate = 'CCCC(=O)OC'
methyl_crotonate = 'CC=CC(=O)OC'
methyl3hexenoate = 'CCC=CCC(=O)OC'
hexene_3 = 'CCC=CCC'
hexadiene_14 = 'C=CCC=CC'
dodecane = 'CCCCCCCCCCCC'
hexadecane = 'CCCCCCCCCCCCCCCC'
decane = 'CCCCCCCCCC'
heptane = 'CCCCCCC'
hexadiene14 = 'C=CCC=CC'
methyl3nonenoate = 'CCCCCC=CCC(=O)OC'
methyldecanoate = 'CCCCCCCCCC(=O)OC'
methyl9decenoate = 'C=CCCCCCCCC(=O)OC'
methyl5decenoate = 'CCCCC=CCCCC(=O)OC'
methyl6decenoate = 'CCCC=CCCCCC(=O)OC'
methyl_oleate = 'CCCCCCCCC=CCCCCCCCC(=O)OC'
methyl_linoleate = 'CCCCCC=CCC=CCCCCCCCC(=O)OC'
methyl_palmiate = 'CCCCCCCCCCCCCCCC(=O)OC'
methyl_stearate = 'CCCCCCCCCCCCCCCCCC(=O)OC'
Gondoic_ester = 'CCCCCCCC=CCCCCCCCCCC(=O)OC'
methyl_linolenate = 'CCC=CCC=CCC=CCCCCCCCC(=O)OC'
methyl10undecenoate = 'C=CCCCCCCCCC(=O)OC'
hexadecane = 'CCCCCCCCCCCCCCCC'

#Composition = [oleate,linoleate,linolenate,palmiate,stearate]
Biodiesel_comp = [methyl_oleate,methyl_linoleate,methyl_linolenate,methyl_palmiate,methyl_stearate]
rapeseed = (0.599,0.211,0.132,0.043,0.013) # westbrook US NREL
RME_pyro = (0.53,0.15,0.05,0.21,0.025) # #Ruben 2015
soybean = (0.388,0.432,0,0.167,0.013) # Ang li 2019
waste_cooking_oil = (0.49, 0.238, 0, 0.237, 0.035)  # Ang li 2019
canola = (0.604,0.212,0.096,0.02,0.042) # Hoekman 2012
Palm = (0.4347, 0.1802, 0, 0.2809, 0.0953) # Hoang 2013
soy_wang = (0.343, 0.226, 0.025, 0.184, 0.088) # Wang 2014
fat_wang = (0.233, 0.523, 0, 0.107, 0.043) # wang 2014

mp26dec74 = {methyl_palmiate:0.26,decane:0.74}
mo26dec74 = {methyl_oleate:0.26,decane:0.74}
soy30hept70 = {soybean:0.3,heptane:0.7}
wco30hept70 = {waste_cooking_oil:0.3,heptane:0.7}
rme30mo70 = {rapeseed:0.3,methyl_oleate:0.7}
md5dmd6d = {methyl5decenoate:0.5,methyl6decenoate:0.5}

#[Cetane_number, LHV, density]
Properties = {methyl_butanoate: [6.4,31.84,0.898],
dodecane:[77.1,44.23,0.745], 
hexene_3:[17.195,44.92,0.678], 
methyl_crotonate: [6.547,31.01,0.944], 
methyldecanoate: [51.6,37.178, 0.8726],
rapeseed : [53.7,38.53,0.879],
soybean : [54.4,37.3,0.8781],
heptane : [52.4,44.56,0.6828],
decane : [69.5,44.35, 0.73098],
methyl_palmiate : [85.7,39.63, 0.8644],
methyl_stearate : [101,40.06, 0.8627],
methyl_oleate : [59.3,39.93, 0.8746],
methyl_linoleate : [38.2,39.65, 0.8865],
Palm : [66.9, 37.39, 0.8836 ],
soy_wang : [59.846,39.31199,0.8737],
fat_wang : [50.7019,38.9467,0.8789],
waste_cooking_oil : [58.7,39.8,0.8750],
methyl10undecenoate : [1.0,1.0,1.0],
methyl3nonenoate : [1.0,1.0,1.0],
methyl5decenoate : [1.0,1.0,1.0],
methyl9decenoate : [1.0,1.0,1.0],
hexadiene14 : [1.0,1.0,1.0],
methyl3hexenoate : [1.0,1.0,1.0],
hexadecane : [1.0,1.0,1.0],
methyl9decenoate : [1.0,1.0,1.0],
methyl6decenoate : [1.0,1.0,1.0],
RME_pyro : [1.0,1.0,1.0],
}

#########################################################################################################################################

list_all = [rapeseed, soybean, soy_wang, Palm, fat_wang, mp26dec74,mo26dec74,soy30hept70,wco30hept70,rme30mo70,methyl_palmiate, methyl_stearate,methyl_oleate,methyl_linoleate,methyl10undecenoate,
methyl3nonenoate,methyl5decenoate,methyldecanoate,methyl9decenoate,md5dmd6d,RME_pyro]

list_str = ['rapeseed', 'soybean', 'soy_wang', 'Palm', 'fat_wang', 'mp26dec74','mo26dec74','soy30hept70','wco30hept70','rme30mo70','methyl_palmiate', 'methyl_stearate','methyl_oleate','methyl_linoleate','methyl10undecenoate',
'methyl3nonenoate','methyl5decenoate','methyldecanoate','methyl9decenoate', "md5dmd6d","RME_pyro"]

surrogate_components = [[methyl_butanoate,dodecane,hexene_3],[methyl_crotonate,dodecane], [methyl_butanoate,dodecane]]

print("List of target fuels: \n", list_str ,"\n")

def formulate(mixture):
	matrix = []
	for i, smile in enumerate(mixture):#
		matrix.append(find_group(smile))  
	return np.asarray(matrix).T

 
index = input('''Enter the 0 for [methyl_butanoate,dodecane,hexene_3] \n
or 1 for [methyl_crotonate,dodecane]c \n 
or 2 for [methyl_butanoate,dodecane] \n
''')

mixture = surrogate_components[int(index)]

fuel = input("Enter name of target fuel from above list  ")

if fuel in list_str:
	Target_fuel = list_all[list_str.index(fuel)]

else:
	print("Enter the valid fuel name")

y = find_target(Target_fuel)

A = formulate(mixture)

x = optmize(A, y)

ans = A @ x
print("Target array:", y,y[6]/y[5],y[7]/y[5])
print("Predicted array:", ans,ans[6]/ans[5],ans[7]/ans[5])


print("Plotting...", plot_analysis(x,A,y))
	  

