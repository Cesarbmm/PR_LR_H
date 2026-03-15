from chromahack.training.train_proxy_cnn import (
    FrameDataset,
    SimpleGradCAM,
    build_parser,
    evaluate,
    fragility_report,
    main,
    run,
    train_one_epoch,
)

__all__ = [
    "FrameDataset",
    "SimpleGradCAM",
    "build_parser",
    "evaluate",
    "fragility_report",
    "main",
    "run",
    "train_one_epoch",
]

if __name__ == "__main__":
    main()
