bash install_requirements.sh
bash run_model.sh train data/train.tsv
bash run_model.sh test test/big_input test/my_big_output
python3 eval.py test/big_output test/my_big_output
