#!/bin/bash

# $1 = input name
# $2 = output name

cd-hit-est -i $1 -o $2 -c 0.95 -n 10 -T 0 -aS .9 -s .9 -M 0 