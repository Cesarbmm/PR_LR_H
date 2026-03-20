from chromahack.intervention.preference_reward_model import (
    LearnedRewardWrapper,
    PreferenceDataset,
    PreferenceRewardModel,
    build_parser,
    collect_trajectories,
    generate_preference_pairs,
    main,
    plot_before_after,
    retrain_with_learned_reward,
    train_preference_model,
)

__all__ = [
    "LearnedRewardWrapper",
    "PreferenceDataset",
    "PreferenceRewardModel",
    "build_parser",
    "collect_trajectories",
    "generate_preference_pairs",
    "main",
    "plot_before_after",
    "retrain_with_learned_reward",
    "train_preference_model",
]

if __name__ == "__main__":
    main()
