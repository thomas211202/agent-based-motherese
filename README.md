# agent-based-motherese
Code for the Agent-based motherese project

Initialise the conda environment by running: 

`conda env create -f env.yml`


Run the training script locally as follows:

`python3 -m src.train\
  --n_experiments 1\
  --prod True\
  --batch_size 32\
  --pretrain_epochs 30\
  --n_epochs 10\
  --vocab_size 8\
  --max_len 3 \
  --sender_hidden 20 \
  --receiver_hidden 20\
  --lr 0.0001\
  --rounds 10\
  --sender_cell "rnn"\
  --receiver_cell "rnn"\
  --n_distractors 7
  `

Alternatively, the experiments can be run on slurm using a jobscript provided in the 
`jobs` folder.
