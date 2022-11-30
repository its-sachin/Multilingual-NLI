if [ "$1" = "train" ]
then
    python3 train_adapter.py $2
    python3 finetune_adapter.py $2
else
    TOKENIZERS_PARALLELISM=true python3 test_model.py $2 $3
fi