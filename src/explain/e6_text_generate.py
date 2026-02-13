from __future__ import annotations

from .family_artifacts import MethodProfile, build_common_parser, run_generator


PROFILE = MethodProfile(
    method_name="text_unconstrained",
    sanity_mu=0.48,
    nuisance_mu=0.66,
    quality_mu=0.52,
    prediction_shift_sigma=0.05,
    concept_drop_prob=0.34,
    concept_add_prob=0.22,
    rationale_flip_prob=0.28,
)


def build_argument_parser():
    return build_common_parser("Generate E6 unconstrained-text proxy artifacts for E7/E8 contract testing.")


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

