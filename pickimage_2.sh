#!/bin/bash


printf 'data/%s 1\n' good/*.jpg >> testtmp.txt
printf 'data/%s 0\n' bad/*.jpg >> testtmp.txt


sort -R testtmp.txt > test.txt
rm testtmp.txt
