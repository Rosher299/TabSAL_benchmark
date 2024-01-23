models="GReaT"
# datasets=("Adult" "Insurance" "Loan" "California" "Buddy" "Abalone" "Diabetes")
# epochs=("300" "300" "300" "300" "300" "300" "300")
datasets=("Diabetes")
epochs=("200")
for model in $models
do
    for i in "${!datasets[@]}"
    do
        dataset=${datasets[$i]}
        typeset -l file_path
        file_path=$model
        echo python ~/Project/ltg_benchmark/baseline/main.py --model=$model --train_or_sample=predict --dataset=$dataset --data_dir=/data/lijiale/data/$dataset/$file_path
        python ~/Project/ltg_benchmark/baseline/main.py --model=$model --train_or_sample=predict  --dataset=$dataset --data_dir=/data/lijiale/data/$dataset/$file_path 
    done
done