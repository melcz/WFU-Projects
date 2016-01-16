#!/bin/bash
n=3000
while [ $n -ge 0 ]
do
 ./MatrixMult "$n"
 n=$(( n-10 ))
done
