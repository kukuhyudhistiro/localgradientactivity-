# Reproducibility Notes

This file records the decisions that affect whether a reader can reproduce the
numbers in the manuscript, including the ones that reflect badly on the method.

## Environment

| Component | Version |
|---|---|
| Processor | Intel Core i5-14400, 16 GB memory |
| Python | 3.13 |
| OpenCV | 4.13 |
| Threading | All library thread counts forced to one |

All timing is single-threaded. The just-in-time compiler is warmed before
timing so that compilation overhead is excluded.

## Corrections applied relative to the submitted version

Five defects were found in the earlier implementation and are corrected here.
They are listed because three of them changed the reported numbers.

**Preprocessing was not uniform.** Canny received the grayscale image before
histogram equalization while the other five methods received the equalized
image. All six now receive the same input.

**Border handling was not uniform.** The two static Gabor baselines used zero
padding while the proposed method used reflection padding. Zero padding creates
spurious responses at the image border and depresses the precision of the
baselines. After correction, the imaginary-only Gabor wavelet rose from 0.406
to 0.417 and the complex Gabor wavelet from 0.392 to 0.397, so the margin of
the proposed method narrowed from 0.128 to 0.117.

**The Canny output was binary.** A binary map collapses a 99-threshold sweep to
a single operating point, which is why average precision was reported as zero
and why the optimal image scale value fell below the optimal dataset scale
value on one dataset. Canny is now evaluated as the Gaussian-smoothed gradient
magnitude before non-maximum suppression, matching the definition used in the
companion repository, and the two implementations agree to within 0.001 ODS.

**The two summary metrics used different aggregation schemes.** The optimal
dataset scale value came from pooled counts while the optimal image scale value
was a mean of per-image F-measures. Both now use dataset-level accumulation.

**The entropy analysis was invalid.** Shannon entropy was computed on
normalised floating-point arrays, in which almost every pixel is a distinct
symbol, so the reported values measured the diversity of floating-point values
rather than the information content of the edge map. That analysis is removed
and replaced by precision at fixed recall, localisation error, and edge width.

## Known limitations of the implementation

The Gabor kernel is not normalised, so its L2 norm varies from 3.010 at the
lower frequency bound to 0.030 at the upper bound. The response magnitude
therefore carries an amplitude scaling correlated with the activity estimate.
The variant `A11_l2norm` in the ablation runner applies normalisation for
readers who wish to measure the effect.

Equation (8) couples scale to frequency through a constant of 0.1874. This is
proportional to, but not identical with, the standard bandwidth relation, which
carries an additional factor of 3 at one octave. The kernel is also evaluated
in a fixed 7 by 7 window, so the envelope is truncated and the fraction of
envelope mass retained falls from 88 percent at the upper bound to 11 percent
at the lower bound. The relation is therefore a scale-frequency coupling rather
than a constant-bandwidth constraint. The variant `A10_octave` applies the full
relation.

The convolution is implemented as a correlation. For the antisymmetric kernel
used here the two differ only in sign, so they are equivalent after the
absolute value is taken.

## Timing measurements

Two earlier timing runs are not usable and are retained only as documentation.
In `results/diagnostics/runtime_grid_raw.csv` the mean drifts by a factor of two
within a single variant on identical images, in opposite directions for
different variants, which indicates background load or thermal throttling
rather than algorithmic variation. A variant that ran while the machine was
quiet has a standard deviation of about 3 percent.

The figures in the manuscript come from `benchmark_runtime.py`, run alone on an
idle machine with three repeats per image and the minimum retained. The
standard deviation for the proposed method is 0.96 percent of the mean.

## Parameter selection

The three sigmoid parameters were verified on the BSDS500 validation split of
100 images. No test image was used at any stage of parameter selection. The
grid covered the lower frequency bound over four values and the sigmoid
steepness over two values. The upper bound was excluded after a separate sweep
showed that varying it from 0.25 to 0.55 changes ODS by 0.0009 in total.

The setting used throughout is within one standard error of the best value on
the grid, so it was retained rather than replaced.
