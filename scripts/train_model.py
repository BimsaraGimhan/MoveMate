"""CLI entrypoint for training."""

from __future__ import annotations

from src.train import main, parse_args


if __name__ == "__main__":
    args = parse_args()
    main(data_path=args.data_path, generate_sample=args.generate_sample)
