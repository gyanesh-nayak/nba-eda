# NBA Shot Probability Modeling

Portfolio overview of a basketball shot probability modeling project completed as part of an internship assessment.

## Project overview

The task was to estimate the probability that a shot is made. The work focused on building a reproducible modeling workflow for probabilistic predictions, with validation designed to measure performance across separate seasons.

## Approach

- Explored outcome patterns and data quality before choosing features.
- Compared a regularized logistic regression baseline with tree-based models.
- Used log loss to evaluate probability estimates.
- Held out complete seasons during validation to check how models transferred across time.
- Kept preprocessing and model fitting together in a pipeline.
- Reviewed calibration and prediction checks as part of the final workflow.

## What I learned

This project strengthened my experience with probability-focused evaluation, season-based validation, categorical preprocessing, tree ensembles, calibration analysis, and communicating model limitations. The most useful lesson was to align validation with the way a model will be used and to avoid interpreting small score differences as meaningful without stronger evidence.

## Data and materials

This repository overview intentionally contains no assessment data, row-level records, derived plots, submission file, detailed assessment results, or employer-provided materials. The internship assessment terms may restrict sharing project code and results as well as the original data. Confirm what may be published with the internship provider before adding those materials.

## Reproduction

A reproducible version is not included because it would require the assessment dataset and may include materials subject to the assessment terms. This page summarizes the general methods and learning outcomes only.
