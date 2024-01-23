models="TVAE"
datasets=("Diabetes")
nums=("1000")
for model in $models
do
    for i in  "${!datasets[@]}"
    do
        dataset=${datasets[$i]}
        num=${nums[$i]}
        # 使用typeset/declare声明变量的属性，将其设置为默认小写
        typeset -l file_path
        file_path=$model
        echo python ~/Project/ltg_benchmark/baseline/main.py --model=$model --train_or_sample=sample --dataset=$dataset --data_dir=./data/$dataset/$file_path --sample_nums=$num
        python ~/Project/ltg_benchmark/baseline/main.py --model=$model --train_or_sample=sample --dataset=$dataset --data_dir=./data/$dataset/$file_path  --sample_nums=$num
    done
done