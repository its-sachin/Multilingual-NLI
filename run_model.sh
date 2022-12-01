if [ "$1" = "train" ]
then
    python3 train_adapter.py $2
    python3 finetune_adapter.py $2
    bash run_model.sh test test/big_input test/my_big_output
    python3 eval.py test/big_output test/my_big_output
    python3 finetune_nli.py $2

else
    TOKENIZERS_PARALLELISM=true python3 test_model.py $2 $3
fi