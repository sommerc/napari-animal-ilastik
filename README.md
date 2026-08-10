# napari-animal-ilastik

A [napari](https://napari.org) plugin for ilastik-like animal behavior classification from SLEAP pose-tracking data.


> ⚠️ Work in progress ⚠️

## Introduction

**napari-animal-ilastik** brings *interactive learning* to behavioral segmentation. You
paint a handful of behavior labels onto bouts in a timeline, a classifier trains on the
fly, and predictions are filled in across your whole dataset - then refine and repeat.

- **Interactive** - annotate on the timeline and get a trained model immediately, no
  train-and-wait cycle.
- **Generic, pose-derived features** - built from any SLEAP skeleton's body-part
  coordinates; nothing species-specific to hand-craft.
- **ilastik-like temporal feature bank** - expands every feature across multiple time
  scales for a larger temporal receptive field, so decisions use context, not single frames.
- **Random Forest classification** - robust with few labels, and reports out-of-bag
  accuracy and a confusion matrix so you can see where the model struggles.
- **Automatable inference** - batch/headless CLI (`anilastik-train`, `anilastik-predict`)
  for whole datasets.
- **No programming expertise required** - everything is point-and-click inside napari.
- **Fast and parallel** - training and prediction use all available CPU cores.

### Kinematic features

Computed per individual from the body-part coordinates in each frame:

- **Distances** - length of each skeleton segment (the distance between connected body parts).
- **Angles** - signed interior angle at every skeleton node with two or more neighbors
  (joints and junctions alike).
- **Angular velocity** - signed rotation rate of each body part around the whole-body centroid.
- **Speed** - frame-to-frame displacement of the centroid and of each individual body part.

### Temporal feature bank

Behavior unfolds over time, not in single frames. Each kinematic channel is expanded with
an ilastik-style multi-scale filter bank applied **along the time axis** at several scales
(σ = 1, 3, 9, 27 frames), giving three responses per scale:

- **Smooth** - Gaussian-smoothed value (the denoised trend at that scale).
- **Rate** - Gaussian first derivative (rate of change / oscillation during a bout).
- **Variability** - Gaussian-windowed standard deviation (local jitter / erratic motion).

This widens the classifier's temporal receptive field - from fast twitches to slow bouts -
while staying strictly per-channel (filters never mix different features). Which
feature-and-scale combinations to use is fully configurable in the **Select features** dialog.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and SSH access to the ISTA GitLab.

Install as an isolated tool (recommended - keeps napari and the plugin in their own
environment, exposes the `napari`, `anilastik-train`, and `anilastik-predict`
commands on your PATH):

```sh
uv tool install "napari-animal-ilastik @ git+ssh://git@git.ista.ac.at/csommer/napari_behavior_classifier.git"
```


## Usage

Launch napari and open the plugin from the menu: **Plugins -> Behavior Classifier**

```sh
napari
```

For batch/headless use:

```sh
anilastik-train session.json model.joblib      # pool a saved session into a model
anilastik-predict model.joblib file1.h5 file2.h5 ...  # predict, one CSV per file
```

## Updating

```sh
uv tool upgrade napari-animal-ilastik
```

## Acknowledgements

Built with the help of [Claude Code](https://claude.com/claude-code). All code is reviewed and validated by the author.

