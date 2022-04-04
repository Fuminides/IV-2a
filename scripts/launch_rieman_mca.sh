#!/bin/bash
#$ -q cal.q
#$ -cwd
#$ -t 1-36
source activate py365

argumentos[${#argumentos[@]}]="main_riemannian.py cost_0_groups_2_test.txt 0 2"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_0_groups_3_test.txt 0 3"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_0_groups_4_test.txt 0 4"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_0_groups_5_test.txt 0 5"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_0_groups_6_test.txt 0 6"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_0_groups_7_test.txt 0 7"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_0_groups_8_test.txt 0 8"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_0_groups_9_test.txt 0 9"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_0_groups_10_test.txt 0 10"

argumentos[${#argumentos[@]}]="main_riemannian.py cost_1_groups_2_test.txt 1 2"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_1_groups_3_test.txt 1 3"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_1_groups_4_test.txt 1 4"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_1_groups_5_test.txt 1 5"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_1_groups_6_test.txt 1 6"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_1_groups_7_test.txt 1 7"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_1_groups_8_test.txt 1 8"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_1_groups_9_test.txt 1 9"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_1_groups_10_test.txt 1 10"

argumentos[${#argumentos[@]}]="main_riemannian.py cost_2_groups_2_test.txt 2 2"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_2_groups_3_test.txt 2 3"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_2_groups_4_test.txt 2 4"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_2_groups_5_test.txt 2 5"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_2_groups_6_test.txt 2 6"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_2_groups_7_test.txt 2 7"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_2_groups_8_test.txt 2 8"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_2_groups_9_test.txt 2 9"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_2_groups_10_test.txt 2 10"

argumentos[${#argumentos[@]}]="main_riemannian.py cost_3_groups_2_test.txt 3 2"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_3_groups_3_test.txt 3 3"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_3_groups_4_test.txt 3 4"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_3_groups_5_test.txt 3 5"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_3_groups_6_test.txt 3 6"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_3_groups_7_test.txt 3 7"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_3_groups_8_test.txt 3 8"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_3_groups_9_test.txt 3 9"
argumentos[${#argumentos[@]}]="main_riemannian.py cost_3_groups_10_test.txt 3 10"

python ${argumentos[$SGE_TASK_ID-1]} 2>error.txt >salida.txt