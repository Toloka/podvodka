git clone https://github.com/huggingface/transformers.git
cp -r transformers/examples/pytorch/language-modeling ./
python3 -m pip install evaluate
python3 -m pip install -r language-modeling/requirements.txt
python3 -m pip install git+https://github.com/huggingface/transformers
python3 language-modeling/run_clm.py --model_name_or_path gpt2 --train_file train_strings.txt --validation_file val_strings.txt --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --learning_rate 5e-04 --preprocessing_num_workers 4 --do_train --do_eval --logging_dir gpt2-finetuned-log --output_dir gpt2-finetuned --num_train_epochs 5 --logging_steps 5 --save_steps 50 --save_total_limit 10 --evaluation_strategy steps --eval_steps 50
