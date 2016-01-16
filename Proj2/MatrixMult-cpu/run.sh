#!/bin/bash
n=2450
while [ $n -ge 0 ]
do
 ./MatrixMult "$n"
 n=$(( n-10 ))
done
