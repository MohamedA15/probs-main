

# Installation

We used conda environment on Ubuntu 20.04.4 LTS.

1). Install conda https://conda.io/projects/conda/en/latest/user-guide/install/index.html

2). Create environment: `conda create --name probs_env -c conda-forge -c pytorch --file requirements.txt`

3). Activate environment: `conda activate probs_env`

Enter this in terminal: cd python_impl_generic

# Usage

## Play chess6x6 interactively

`python go_probs.py --cmd play_chess --env mychess6x6`




## Train Chess6x6

1). Create checkpoints directory: `mkdir ~/checkpoints_chess6x6/`

2). Test training loop once:

```
time python -u go_probs.py \
--env mychess6x6 \
--cmd train \
--device gpu \
--sub_processes_cnt 2 \
--self_play_threads 1 \
--n_high_level_iterations 1 \
--v_train_episodes 10 \
--q_train_episodes 10 \
--q_dataset_episodes_sub_iter 10 \
--dataset_drop_ratio 0.1 \
--model V=ValueModel66_v11,SL=SelfLearningModel66_v11 \
--checkpoints_dir=. \
--value_lr 0.0003 \
--self_learning_lr 0.0003 \
--value_batch_size 16 \
--self_learning_batch_size 16 \
--n_max_episode_steps 100 \
--num_q_s_a_calls 10 \
--max_depth 100 \
--get_q_dataset_batch_size 300 \
--alphazero_move_num_sampling_moves 20 \
--enemy two_step_lookahead \
--evaluate_n_games 10
```


## Train Chess8x8

Test training loop once:
```
time python -u go_probs.py \
--env mychess8x8 \
--cmd train \
--device gpu \
--sub_processes_cnt 10 \
--self_play_threads 1 \
--n_high_level_iterations 1 \
--v_train_episodes 60 \
--q_train_episodes 30 \
--q_dataset_episodes_sub_iter 30 \
--dataset_drop_ratio 0.75 \
--model V=ValueModel88_v1,SL=SelfLearningModel88_v1 \
--value_lr 0.0003 \
--self_learning_lr 0.0003 \
--value_batch_size 256 \
--self_learning_batch_size 256 \
--n_max_episode_steps 200 \
--num_q_s_a_calls 10 \
--max_depth 100 \
--get_q_dataset_batch_size 300 \
--alphazero_move_num_sampling_moves 20 \
--enemy two_step_lookahead \
--evaluate_n_games 10 2>&1 | tee training.log
```



## Battle play trained chess agent

1). Trained agent vs random:

`python go_probs.py --cmd battle --env mychess6x6 --model V=ValueModel66_v11,SL=SelfLearningModel66_v11,CKPT=environments/mychess6x6_v11_checkpoint_20240822-132834.ckpt --enemy random --evaluate_n_games 100`


2). Trained agent vs two_step_lookahead:

`python go_probs.py --cmd battle --env mychess6x6 --model V=ValueModel66_v11,SL=SelfLearningModel66_v11,CKPT=environments/mychess6x6_v11_checkpoint_20240822-132834.ckpt --enemy two_step_lookahead --evaluate_n_games 100`


  