###annotate a gene catalogue with biocyc
##20180315
#Tierney

#DEPENDENCIES: numpy, click, pandas, diamond 
#run python annotate.py --help to see options 

import numpy as np
import os
import sys 
import click
import pandas as pd
import sys
from sys import argv
import subprocess
from subprocess import check_output

def parse_dictionaries(dictionary,genes_from_alignments,products_from_alignments,gene_prod_mapping,enz):
	output=[]
	for i in range(0,len(genes_from_alignments)):
		query=genes_from_alignments[i]
		try:
			a=gene_prod_mapping[products_from_alignments[i]][0]
			if enz==True:
				path=dictionary[a][0]
			if enz==False:
				path=dictionary[a]
			path=[query]+[a]+path
			if len(path)>2:
				output.append(['	'.join(path)])
		except Exception as e: 
			continue
	return output

def load_annotated_data(dictname):
	outputDict = {}
	vals=[]
	keys=[]
	futureindex=[]
	with open(dictname) as f:
		for line in f:
			try:
				key = line.split('\t')[0].rstrip()
				if 'transporter' in dictname:
					val = line.split('\t')[3].rstrip()
				else:
					val = line.split('\t')[2].rstrip()
				try:
					outputDict.setdefault(key, []).append(val)
				except:
					outputDict[key] = [val]
				futureindex.append(val)
			except Exception as e:
				continue
	futureindex=[x.split('|') for x in futureindex]
	futureindex=[item for sublist in futureindex for item in sublist]
	return [outputDict,list(set(futureindex))]

def collapse_annotations(abMat,dictIndexPairs,abmat_delimiter):
	annoCountDict={}
	outputDFs=[]
	f=open(abMat)
	for i,line in enumerate(f):
		if i==0:
			newCols=line.rstrip().split(abmat_delimiter)[1:]
			for d in dictIndexPairs:
				outputDFs.append(pd.DataFrame(0.0, index=d[1], columns=newCols))
			continue
		gene=line.split('\t')[0]
		line=line.rstrip().split('\t')[1:]
		for j,x in enumerate(line):
			if x == '':
				line[j]='0.0'
		line=[float(x) for x in line]
		data=pd.DataFrame([np.array(line,dtype=float)],index=[gene],columns=newCols)
		for ii,d in enumerate(dictIndexPairs):
			outputDF=outputDFs[ii]
			annoDict=d[0]
			try:
				annotations=annoDict[gene]
			except:
				continue
			for anno in annotations:
				anno2=anno.split('|')
				for anno3 in anno2:
					outputDF.loc[anno3,:]=outputDF.loc[anno3,:]+data.iloc[0,:]
					try:
						annoCountDict[anno3]=annoCountDict[anno3]+1
					except:
						annoCountDict[anno3]=1
			outputDFs[ii]=outputDF
	for iiii,outputDF in enumerate(outputDFs):
		for iii in range(0,len(outputDF.index)):
			try:
				outputDF.iloc[iii,:]=outputDF.iloc[iii,:]/float(annoCountDict[outputDF.index[iii]])
			except:
				continue
		outputDFs[iiii]=outputDF
	return outputDFs

@click.command()
@click.option('--catalog', '-g',help='Path to your fasta file to annotate (default: None)',default=False)
@click.option('--abmat', '-a',help='Path to your abundance matrix (default: None)',default=False)
@click.option('--existing_annotations', '-e',help='If only collapsing abundance matrix, add comma-separated paths to any and all existing annotations (i.e. ~/biocyc_complex_annotation,./biocyc_pathway_annotation).',default='biocyc_pathway_annotation,biocyc_complex_annotation,biocyc_transporter_annotation,biocyc_enzymatic_annotation')
@click.option('--annotation_databases', '-d',help='Add path to folder in which databases are stored if they are not in the working directory (i.e. ~/biocyc_db).',default='./')
@click.option('--diamond_path', '-i',help='Path to diamond installation if it is not in the home directory (i.e. ~/programs/diamond)',default='~/diamond')
@click.option('--numalign','-n', default=20, help='Number of alignments allowed per gene (default: 20)')
@click.option('--seqtype', '-s',help='Specify p or n for protein or nucleic acid sequence, respectively (default: p)',default='p')
@click.option('--threads', '-t',help='Cores to be used (default: 4)',default=4)
@click.option('--abmat_delimiter', '-l',help='Delimiter used in your abundance matrix (default: \t)',default='\t')


def main(catalog,numalign,seqtype,threads,abmat,existing_annotations,annotation_databases,diamond_path,abmat_delimiter):
	if catalog==False and abmat==False:
		print "You don't seem to have specified any data to process...either input an fasta file, abundance matrix (that has row names corresponding to an annotated fasta file), or both. Otherwise...well I'm not sure why you're doing this. Thanks for pulling my code, though!"
		return None
	annotate=True
	if catalog==False:
		annotate=False
		print 'No sequences to annotate given, attempting to use existing annotated data.'
	#run diamond
	if annotate==True:
		if seqtype=='n':
			diamondtype='blastx'
		else: 
			diamondtype='blastp'
		os.system("%s %s --id 95 --threads %s -d biocyc_db -k %s -q %s -o %s_diamond_out"%(diamond_path,diamondtype,threads,numalign,catalog,catalog))

		#load and parse alignments
		alignments=[]
		genes=[]
		f=open('%s_diamond_out'%catalog)
		for line in f:
			genes.append(line.split('\t')[0])
			alignments.append(''.join(line.split('\t')[1].split('|')[-1].split('_')[:-1]))

		#load in dictionaries
		gene_product={}
		f=open('biocyc_flatfile_gene_product')
		for line in f:
			gene_product[line.split('\t')[0]]=line.rstrip().split('\t')[1:]

		product_path={}
		f=open('biocyc_flatfile_product_path')
		for line in f:
			product_path[line.split('\t')[0]]=line.rstrip().split('\t')[1:]

		print 'Finding pathway annotations...'
		out=parse_dictionaries(product_path,genes,alignments,gene_product,False)
		w=open('biocyc_pathway_annotation','w')
		for line in out:
			w.write(''.join(line)+'\n')

		print 'Finding enzymatic and reaction annotations...'
		product_enz={}
		f=open('biocyc_flatfile_product_enzyme')
		for line in f:
			product_enz[line.split('\t')[0]]=[line.rstrip().split('\t')[1:]]

		out=parse_dictionaries(product_enz,genes,alignments,gene_product,True)
		w=open('biocyc_enzymatic_annotation','w')
		for line in out:
			w.write(''.join(line)+'\n')

		print 'Finding transporter annotations...'
		product_trans={}
		f=open('biocyc_flatfile_product_transporter')
		for line in f:
			product_trans[line.split('\t')[0]]=line.rstrip().split('\t')[1:]

		out=parse_dictionaries(product_trans,genes,alignments,gene_product,False)
		w=open('biocyc_transporter_annotation','w')
		for line in out:
			w.write(''.join(line)+'\n')

		print 'Finding protein complex annotations...'
		product_complex={}
		f=open('biocyc_flatfile_product_complex')
		for line in f:
			product_complex[line.split('\t')[0]]=line.rstrip().split('\t')[1:]

		out=parse_dictionaries(product_complex,genes,alignments,gene_product,False)
		w=open('biocyc_complex_annotation','w')
		for line in out:
			w.write(''.join(line)+'\n')

		print 'Annotation process complete.'

	if abmat is None:
		return 'Completed, as no abundance matrix was given as input.'

	print 'Collapsing annotation data onto abundance matrix...'
	annotations=existing_annotations.split(',')
	annotationPairs=[]
	for annotationDict in annotations:
		print 'Loading %s data...'%annotationDict
		annotationPairs.append(load_annotated_data(annotationDict))
	print 'Merging annotations and abundance matrix...'
	output=collapse_annotations(abmat,annotationPairs,abmat_delimiter)
	print 'Writing output...'
	for i,o in enumerate(output):
		o.to_csv(annotations[i].split('_')[1]+'_annotation_abMat.csv')

	print 'Done!'
	return None
if __name__ == "__main__":
    main()

