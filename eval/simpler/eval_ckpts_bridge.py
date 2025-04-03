import os

ckpt_paths = [
    # (
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_kosmos/2025-03-19/00-21/epoch-4.pt",
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_kosmos/2025-03-19/00-21/config.json",
    # ),
    # (
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_paligemma/2025-03-19/00-02/epoch2.pt",
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_paligemma/2025-03-19/00-02/config.json"
    # )
    # (
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_kosmos/2025-03-19/00-21/epoch-9.pt",
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_kosmos/2025-03-19/00-21/config.json",
    # ),
    # (
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_paligemma/2025-03-31/20-17/epoch-2_step-50k.pt",
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_paligemma/2025-03-31/20-17/config.json"
    # )
    # (
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_paligemma/2025-03-31/20-17/epoch-4_step-70k.pt",
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_paligemma/2025-03-31/20-17/config.json"
    # )
    # (
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_paligemma/2025-04-03/00-37/epoch-2_5k-move_gripper-bs_1024.pt",
    #     "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_paligemma/2025-04-03/00-37/config.json"
    # ),
    (
        "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_paligemma/2025-03-31/20-17/epoch-4_step-70k.pt",
        "/share/project/lxh/project/QL/RoboVLMs/runs/bridge_finetune_paligemma/2025-03-31/20-17/config.json"
    )
]

for i, (ckpt, config) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.system("bash scripts/bridge.bash {} {}".format(ckpt, config))
