#!/bin/bash
executable=$1
size=$2
rm _tmp.log
touch _tmp.log
for i in {1..10}
do
	./$executable $size >> _tmp.log
done
python average.py
