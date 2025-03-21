import os

ckpt_paths = [
    # (
    #     "runs/checkpoints/paligemma/libero_finetune/2025-02-12/11-39/epoch=1-step=40912.ckpt",
    #     "runs/logs/paligemma/libero_finetune/2025-02-12/11-39/2025-02-12_11:39:32.253346-project.json",
    # ),
    # (
    #     "runs/checkpoints/paligemma/libero_finetune/2025-01-13/07-17/epoch=1-step=33822.ckpt",
    #     "runs/logs/paligemma/libero_finetune/2025-01-13/07-17/2025-01-13_07:17:10.495786-project.json",
    # ),
    # (
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/checkpoints/paligemma/libero_finetune_grpo/2025-03-06/14-21/epoch=0-step=10000.ckpt",
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/logs/paligemma/libero_finetune_grpo/2025-03-06/14-21/2025-03-06_14:22:25.450351-project.json",
    # ),
    # (
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/checkpoints/paligemma/libero_finetune_grpo/2025-03-06/14-21/epoch=0-step=20000.ckpt",
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/logs/paligemma/libero_finetune_grpo/2025-03-06/14-21/2025-03-06_14:22:25.450351-project.json",
    # ),
    # (
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/checkpoints/paligemma/libero_finetune_grpo/2025-03-06/14-21/epoch=0-step=30000.ckpt",
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/logs/paligemma/libero_finetune_grpo/2025-03-06/14-21/2025-03-06_14:22:25.450351-project.json",
    # ),
    # (
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/checkpoints/paligemma/libero_finetune_grpo/2025-03-06/14-21/epoch=0-step=40000.ckpt",
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/logs/paligemma/libero_finetune_grpo/2025-03-06/14-21/2025-03-06_14:22:25.450351-project.json",
    # ),
    # (
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/checkpoints/paligemma/libero_finetune_grpo/2025-03-06/14-21/epoch=0-step=50000.ckpt",
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/logs/paligemma/libero_finetune_grpo/2025-03-06/14-21/2025-03-06_14:22:25.450351-project.json",
    # ),
    # (
    #     "/share/project/lxh/CKPT/2025-03-04-01-12/epoch-0_step-10000.pt",
    #     "/share/project/lxh/CKPT/2025-03-04-01-12/config.json",
    # )
    # (
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/checkpoints/paligemma/libero_finetune_cot/2025-03-10/11-26/epoch=0-step=30000.ckpt",
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/logs/paligemma/libero_finetune_cot/2025-03-10/11-26/2025-03-10_11:27:03.785883-project.json"
    # )
    # (
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/checkpoints/paligemma/libero_finetune_cot/2025-03-10/11-26/epoch=0-step=10000.ckpt",
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/logs/paligemma/libero_finetune_cot/2025-03-10/11-26/2025-03-10_11:27:03.785883-project.json"
    # )
    # (
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/checkpoints/paligemma/libero_finetune_verifycot/2025-03-18/18-45/epoch=9-step=3990.ckpt",
    #     "/share/project/lxh/project/RoboVLMs-CoT/runs/logs/paligemma/libero_finetune_verifycot/2025-03-18/18-45/2025-03-18_18:46:31.651810-project.json"
    # )
    (
        "/share/project/lxh/project/RoboVLMs-CoT/runs/checkpoints/paligemma/libero_finetune_verifycot/2025-03-18/18-45/epoch=3-step=1596.ckpt",
        "/share/project/lxh/project/RoboVLMs-CoT/runs/logs/paligemma/libero_finetune_verifycot/2025-03-18/18-45/2025-03-18_18:46:31.651810-project.json"
    )
]

for i, (ckpt, config) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.system("bash scripts/run_eval_libero_lxh.sh {} {}".format(ckpt, config))
