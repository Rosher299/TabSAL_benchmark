models="TVAE"
# datasets=("Adult" "Insurance" "Loan" "California" "Buddy" "Abalone" "Diabetes")
# epochs=("300" "300" "300" "300" "300" "300" "300")
datasets=("Diabetes")
epochs=("200")
for model in $models
do
    for i in "${!datasets[@]}"
    do
        dataset=${datasets[$i]}
        epoch=${epochs[$i]}
        typeset -l file_path
        file_path=$model
        echo python ~/Project/ltg_benchmark/baseline/main.py --model=$model --train_or_sample=train --dataset=$dataset --data_dir=./data/$dataset/$file_path --epoch=$epoch
        python ~/Project/ltg_benchmark/baseline/main.py --model=$model --train_or_sample=train  --dataset=$dataset --data_dir=./data/$dataset/$file_path --epoch=$epoch 
    done
done