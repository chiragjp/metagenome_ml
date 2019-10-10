#!/bin/bash
###prep normalized matrix for CAGS

echo 'Removing commas...'
sed 's/,/\t/g' $1 > $2
echo 'Removing first line...'
tail -n +2 $2 > temp
mv temp $2
