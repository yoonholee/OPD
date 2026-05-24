# Lessons

- `Qwen/Qwen3.5-2B` is a VLM-style repo (`AutoModelForImageTextToText`, `qwen3_5`, vision config), not a plain text CausalLM. THUNLP/OPD's frozen Transformers 4.x stack does not recognize `qwen3_5`; Transformers main and vLLM nightly do.
- THUNLP/OPD paper defaults: GRPO uses Qwen3-4B-Base, DAPO-Math-17K processed, 1 epoch, global/mini batch 64, n=8, prompt 1024, response 7168, val 31744, lr 1e-6, temp/top-p 1.0, no KL, token-mean. OPD uses global/mini batch 64, n=4, top-k 16, student top-k, top-p 1.0, prompt 1024, response 7168, lr 1e-6, 1 epoch, KL 0.
- THUNLP/OPD on GCP 4xL4 with `use_remove_padding=True` killed Ray workers during FSDP/vLLM init even for Qwen3-0.6B. Clean 4-GPU NCCL all-reduce passed, and disabling remove-padding made GRPO and OPD complete. Inferred root cause: remove-padding or FA2/rmpad path, not basic NCCL.
- Successful 4xL4 smoke: THUNLP GRPO, Qwen3-0.6B, 1 step, `use_remove_padding=False`, max_model_len/max_num_batched_tokens 1024. Evidence: `agent_notes/gcp_runs/20260524_182510_l4_final/thunlp_grpo_qwen3_0p6b_no_remove_padding_1024.log`, rc 0, throughput 24.97 tok/s, max GPU alloc 5.81 GB.
- Successful 4xL4 smoke: THUNLP OPD same-weight teacher/student, Qwen3-0.6B, 1 step, `use_remove_padding=False`. Evidence: `agent_notes/gcp_runs/20260524_182510_l4_final/thunlp_opd_qwen3_0p6b_no_remove_padding_1024.log`, rc 0, throughput 27.33 tok/s, top-k overlap 1.0, actor grad norm 7.7e-7.
- Same weights can still produce material logprob drift across kernels. HF SDPA vs FA2 on Qwen3-1.7B had shared-topk max logprob diff up to 0.75 on a tiny n=4 prompt probe. Evidence: `agent_notes/gcp_runs/20260524_175853_l4/prob_diff_qwen3_l4.log`.
- vLLM nightly can load Qwen3.5-2B on L4 if `gdn_prefill_backend=triton` is passed. Without that, first load triggered long FlashInfer GDN JIT on H100.

- Qwen3.5-2B works in THUNLP/OPD with minimal shims on 4xL4 when using Transformers main and vLLM nightly. Required runtime knobs: `use_remove_padding=False`, `enable_activation_offload=False`, `actor/ref.use_torch_compile=False`, `gpu_memory_utilization=0.45`, `max_num_seqs=32`, image/video multimodal limits set to 0, GDN prefill backend `triton`. Evidence: `agent_notes/gcp_runs/opd-qwen35-2b-l4-0524-1159-final/`.
- Qwen3.5 in vLLM has Mamba cache constraints: a tiny smoke with `max_num_seqs=1024` failed even at `max_model_len=512`; lowering to 32 fixed init. Evidence: retry3 log in final GCP run.
- THUNLP activation offload is unsafe with Qwen3.5/Transformers-main checkpointing here: backward hit `activation_offload.py` tuple assertion. Disabling activation offload made GRPO and OPD pass. Evidence: retry4 failed, retry5 passed.
- vLLM nightly moved `LoRAModel` out of `vllm.lora.models`; compat import must fall back to `vllm.lora.worker_manager` or `vllm.lora.lora_model`.
- Stable Qwen3.5-2B OPD smoke env: `transformers==5.9.0`, `vllm==0.21.0`, Torch `2.11.0+cu130`, NumPy `2.3.5`, Ray `2.55.1` from `local/qwen35_2b_smoke/uv.lock`; no nightly wheels needed.
- Qwen3.5 text-only batching needs `data.return_multi_modal_inputs=False`. Otherwise VLM `multi_modal_inputs` have variable shapes and fail batch collation at batch >1.
- GCP GPU NIC names differ by shape: observed G2/L4 `ens7`, G4 `ens3`, A2/A100 `ens8`. Use route autodetect for `NCCL_SOCKET_IFNAME`.
- GCP G4 `g4-standard-192` requires `hyperdisk-balanced`; `pd-balanced` VM create fails.
- vLLM 0.21 Qwen3.5 colocated rollout can hang in FlashInfer sampler JIT; `VLLM_USE_FLASHINFER_SAMPLER=0` avoids that path, but GDN/FLA Triton warmup still makes step 1 slow.
- THUNLP OPD reward entropy must use `reshape`, not `view`, because Qwen3.5/Transformers 5 can return non-contiguous logits.
- Qwen3.5 remove-padding no longer hard-crashes with PyTorch padding-helper fallback; 3-step G4 probe passed, but throughput is not apples-to-apples yet.
