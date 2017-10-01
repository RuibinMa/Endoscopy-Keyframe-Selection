#!/bin/bash

printf 'data/%s 1\n' good/*.jpg >> traintmp.txt
printf 'data/%s 0\n' bad/*.jpg >> traintmp.txt

sort -R traintmp.txt > train.txt
rm traintmp.txt


printf 'data/%s 1\n' vadgood/*.jpg >> testtmp.txt
printf 'data/%s 0\n' vadbad/*.jpg >> testtmp.txt


sort -R testtmp.txt > test.txt
rm testtmp.txt
