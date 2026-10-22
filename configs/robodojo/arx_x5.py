"""RoboDojo ARX-X5 simulator and ZMQ interface, without model dependencies.

Matches FluxVLA's pi05_paligemma_robodojo_full_finetune.py evaluation:
54 task variants / 2,100 episodes; three RGB cameras, 14D joint state and
14D joint actions, executing at most 50 actions per prediction.
Model weights, tokenizer, normalization and inference settings live in VLA.
ROBODOJO_ROOT selects the installed simulator checkout.
"""

_GENERALIZATION_TASKS = ('stack_bowls', 'push_T', 'pack_objects_into_box',
                         'fold_clothes', 'hang_mugs', 'sweep_blocks',
                         'pour_liquid_into_cup', 'make_toast',
                         'arrange_largest_number',
                         'sort_nesting_dolls_by_size',
                         'store_laptop_and_headphones', 'stack_blocks')

themis = dict(
    transport=dict(
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['states'],
        unnorm_key='robodojo_arx_x5',
    ),
    evaluation=dict(
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='all',
            env_cfg_type='arx_x5',
            robodojo_root=None,
            action_mode='joint',
            headless=True,
            save_videos=False),
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        evaluator=dict(type='SuccessRateEvaluator'),
        base_seed=0,
        episodes_per_task=50,
        episodes_per_task_overrides={
            name: 25
            for task in _GENERALIZATION_TASKS
            for name in (task, f'{task}_random')
        },
        max_episode_steps=2000,
        execute_horizon=50,
        stop_on_success=True,
    ),
)
