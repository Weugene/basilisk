#!/bin/bash

#PBS -l nodes=1:ppn=10

#PBS -l pmem=10000MB

#PBS -l walltime=40:00:00

#PBS -q batch

cd $HOME/basilisk/work/2phase_new/staggered
mpirun -np 10 ./a.out
