models="Tabula GReaT"
# datasets=("Adult" "Insurance" "Loan" "California" "Buddy" "Abalone" "Diabetes")
datasets=("Insurance")
for model in $models
do
    for i in  "${!datasets[@]}"
    do
        dataset=${datasets[$i]}
        num=${nums[$i]}
        # 使用typeset/declare声明变量的属性，将其设置为默认小写
        typeset -l file_path
        file_path=$model
        echo python /home/ljl/Project/be_great/baseline/main.py --model=$model --train_or_sample=distance --dataset=$dataset --data_dir=/data/lijiale/data/$dataset/$file_path
        python /home/ljl/Project/be_great/baseline/main.py --model=$model --train_or_sample=distance --dataset=$dataset --data_dir=/data/lijiale/data/$dataset/$file_path
    done
done