from __future__ import annotations

from .family_artifacts import MethodProfile, build_common_parser, run_generator


PROFILE = MethodProfile(
    method_name="text_constrained",
    sanity_mu=0.31,
    nuisance_mu=0.74,
    quality_mu=0.62,
    prediction_shift_sigma=0.04,
    concept_drop_prob=0.20,
    concept_add_prob=0.08,
    rationale_flip_prob=0.10,
)


def build_argument_parser():
    return build_common_parser("Generate E5 constrained-text proxy artifacts for E7/E8 contract testing.")


def main() -> None:
    args = build_argument_parser().parse_args()
    run_generator(
        args=args,
        profile=PROFILE,
        include_text_columns=True,
        include_concept_intervention=False,
    )


if __name__ == "__main__":
    main()

