set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export DATASETS="../datasets/hub"
export MODEL_PATH="../hub/Qwen2.5-3B-Instruct"
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=4025943f5c98398d235eae04243f882b45bcd591

nnodes=1
n_gpus_per_node=8
total_epochs=50
use_remove_padding=False
use_hidden_states=True
tensor_model_parallel_size=2
project_name='verl_diversity'
model_name=${MODEL_PATH}
experiment_name='verl_Qwen2.5-3B-Instruct_BYTED_GRPO_DV'


mm_train_path=$DATASETS/geometry3k/train.parquet
mm_test_path=$DATASETS/geometry3k/test.parquet
gsm8k_train_path=$DATASETS/gsm8k/train.parquet
gsm8k_test_path=$DATASETS/gsm8k/test.parquet
dapomath_train_path=$DATASETS/DAPO-Math-17k/train.parquet
amie2024_test_path=$DATASETS/AIME-2024/train.parquet

train_multimodal_parquet_files="['$mm_train_path']"
train_text_parquet_files="['$dapomath_train_path']"
val_multimodal_parquet_files="['$mm_test_path']"
val_text_parquet_files="['$amie2024_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_dv \
    data.train_text_parquet_files="$train_text_parquet_files" \
    data.val_text_parquet_files="$val_text_parquet_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.image_key=images \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.model.use_hidden_states=${use_hidden_states} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_epochs $@
