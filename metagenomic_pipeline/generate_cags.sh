#!/bin/bash

#generate cags

# $1 =  number of threads
# $2 =  input normalized abundance file
# $3 =  output name of cluster file
# $4 =  output name of cluster profiles

cc.bin -n $1 -i $2 -o $3 -c $4 
