#!/bin/bash
set -x
set -o nounset
echo "Usage: Tcyl Tin maxlevel dr dx dy"

Tcyl=$1
Tin=$2
maxlevel=$3
ratio_Rbmin=0.166666
ratio_Rbmax=1.
ratio_dist_x=$4
ratio_dist_y=$5
ratio_front_x=$6
cyl_x=$7
Nb=0
Ncx=$8
Ncy=$9
TOLERANCE_P=4e-7
TOLERANCE_V=4e-6
TOLERANCE_T=1e-6
Htr=355000
Arrhenius_const=80600
Ea_by_R=7697 #already divided by 8.31
dev_r=${10}
dev_x=${11}
dev_y=${12}
shift_x=${13}
shift_y=${14}
non_saturated=${15}
mode=${16}


tmp="Tcyl=${Tcyl}_Tin=${Tin}_maxl=${maxlevel}_rdx=${ratio_dist_x}_rdy=${ratio_dist_y}_rfx=${ratio_front_x}_Ncx=${Ncx}_Ncy=${Ncy}_dr=${dev_r}_dx=${dev_x}_dy=${dev_y}_order=${shift_y}_nonsat=${non_saturated}"
mkdir ${tmp} || continue
cd ${tmp} || continue
cp ../a.out .

#find out iter_fp
#iter_fp=$(grep "iter_fp=" log | tail -1 | awk '{print $4}')

args="${Tcyl} ${Tin} ${maxlevel} ${ratio_Rbmin} ${ratio_Rbmax} ${ratio_dist_x} ${ratio_dist_y} ${ratio_front_x} \
${cyl_x} ${Nb} ${Ncx} ${Ncy} ${TOLERANCE_P} ${TOLERANCE_V} ${TOLERANCE_T} \
${Htr} ${Arrhenius_const} ${Ea_by_R} ${dev_r} ${dev_x} ${dev_y} ${shift_x} ${shift_y} ${non_saturated}"

echo "args:${args}"

echo "BASILISK=$BASILISK"

#save last dump into restart
dump_files=($(ls -1v dump-*))
N_dumps=${#dump_files[@]}
if (( $N_dumps > 0)); then
	last_file=${dump_files[${N_dumps} - 1]}
fi
if (( $N_dumps > 0 )); then
  echo "copying ${last_file} to restart ..."
  cp  ${last_file} restart
fi

echo "BASILISK=$BASILISK"
#run the code
if [[ "$mode" == "qsub" ]] ; then
  mpirun ./a.out ${args} >>log 2>&1
else
  srun ./a.out ${args} >>log 2>&1
fi
