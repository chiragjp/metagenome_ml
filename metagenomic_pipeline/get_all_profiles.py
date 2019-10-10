import subprocess
import os
import pandas as pd
import numpy as np

#argument 1 = path to location of all metaphlan output files (you cannot have anything else in this directory)

def build_data_frame(taxalevel_set,all_data,level):
	taxalevel_profile=pd.DataFrame(np.zeros(len(taxalevel_set)),index=taxalevel_set)
	for i in range(0,len(all_data)):
		chunk=all_data[i]
		df=pd.DataFrame(chunk)
	        if len(df.index)==0:
	                continue
		df.index=df.iloc[:,0]
		df.drop(0,axis=1,inplace=True)
		df.columns=[vals1[i].split('_')[0]]
		taxalevel_profile=taxalevel_profile.join(df)
	taxalevel_profile=taxalevel_profile.iloc[:,1:]
	taxalevel_profile=taxalevel_profile.fillna(0)
	taxalevel_profile.to_csv('./%s_profile.csv'%(level))

print 'Loading data...'
vals1=os.listdir(sys.argv[1])
vals1=[sys.argv[1]+'/'+x for x in vals1]

kingdoms_all=[]
phyla_all=[]
classes_all=[]
orders_all=[]
families_all=[]
genera_all=[]
species_all=[]
taxa_all=[]
kingdoms_names=[]
phyla_names=[]
classes_names=[]
orders_names=[]
families_names=[]
genera_names=[]
species_names=[]
taxa_names=[]
for val in vals1:
	kingdoms=[]
	phyla=[]
	classes=[]
	orders=[]
	families=[]
	genera=[]
	species=[]
	taxa=[]
	f=open(val)
	for i,line in enumerate(f):
		if i==0:
			continue
		if len(line.rstrip().split('\t')[0].split('|'))==1:
			kingdoms.append([line.rstrip().split('\t')[0].split('|')[-1][3:],line.rstrip().split('\t')[-1]])
			kingdoms_names.append(line.rstrip().split('\t')[0].split('|')[-1][3:])
		if len(line.rstrip().split('\t')[0].split('|'))==2:
			phyla.append([line.rstrip().split('\t')[0].split('|')[-1][3:],line.rstrip().split('\t')[-1]])
			phyla_names.append(line.rstrip().split('\t')[0].split('|')[-1][3:])
		if len(line.rstrip().split('\t')[0].split('|'))==3:
			classes.append([line.rstrip().split('\t')[0].split('|')[-1][3:],line.rstrip().split('\t')[-1]])
			classes_names.append(line.rstrip().split('\t')[0].split('|')[-1][3:])
		if len(line.rstrip().split('\t')[0].split('|'))==4:
			orders.append([line.rstrip().split('\t')[0].split('|')[-1][3:],line.rstrip().split('\t')[-1]])
			orders_names.append(line.rstrip().split('\t')[0].split('|')[-1][3:])
		if len(line.rstrip().split('\t')[0].split('|'))==5:
			families.append([line.rstrip().split('\t')[0].split('|')[-1][3:],line.rstrip().split('\t')[-1]])
			families_names.append(line.rstrip().split('\t')[0].split('|')[-1][3:])
		if len(line.rstrip().split('\t')[0].split('|'))==6:
			genera.append([line.rstrip().split('\t')[0].split('|')[-1][3:],line.rstrip().split('\t')[-1]])
			genera_names.append(line.rstrip().split('\t')[0].split('|')[-1][3:])
		if len(line.rstrip().split('\t')[0].split('|'))==7:
			species.append([line.rstrip().split('\t')[0].split('|')[-1][3:],line.rstrip().split('\t')[-1]])
			species_names.append(line.rstrip().split('\t')[0].split('|')[-1][3:])
		if len(line.rstrip().split('\t')[0].split('|'))==8:
			taxa.append([line.rstrip().split('\t')[0].split('|')[-1][3:],line.rstrip().split('\t')[-1]])
			taxa_names.append(line.rstrip().split('\t')[0].split('|')[-1][3:])
	kingdoms_all.append(kingdoms)
	phyla_all.append(phyla)
	classes_all.append(classes)
	orders_all.append(orders)
	families_all.append(families)
	genera_all.append(genera)
	species_all.append(species)
	taxa_all.append(taxa)

kingdoms_set=list(set(kingdoms_names))
phyla_set=list(set(phyla_names))
classes_set=list(set(classes_names))
orders_set=list(set(orders_names))
families_set=list(set(families_names))
genera_set=list(set(genera_names))
species_set=list(set(species_names))
taxa_set=list(set(taxa_names))

print 'Catting kingdoms data...'
build_data_frame(kingdoms_set,kingdoms_all,'kingdoms')
print 'Catting phyla data...'
build_data_frame(phyla_set,phyla_all,'phyla')
print 'Catting class data...'
build_data_frame(classes_set,classes_all,'classes')
print 'Catting order data...'
build_data_frame(orders_set,orders_all,'orders')
print 'Catting family data...'
build_data_frame(families_set,families_all,'families')
print 'Catting genus data...'
build_data_frame(genera_set,genera_all,'genera')
print 'Catting species data...'
build_data_frame(species_set,species_all,'species')
print 'Catting sub-species data...'
build_data_frame(taxa_set,taxa_all,'taxa')