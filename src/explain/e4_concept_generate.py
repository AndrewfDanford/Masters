from __future__ import annotations

from .family_artifacts import MethodProfile, build_common_parser, run_generator


PROFILE = MethodProfile(
    method_name="concept_cbm",
    sanity_mu=0.23,
    nuisance_mu=0.80,
    quality_mu=0.70,
    prediction_shift_sigma=0.03,
    concept_drop_prob=0.12,
    concept_add_prob=0.06,
    rationale_flip_prob=0.08,
)


def build_argument_parser():
    return build_common_parser("Generate E4 concept-family proxy artifacts for E7/E8 contract testing.")


def main() -> None:
    args = build_argument_parser().parse_args()
    run_generator(
        args=args,
        profile=PROFILE,
        include_text_columns=False,
        include_concept_intervention=True,
    )


if __name__ == "__main__":
    main()

