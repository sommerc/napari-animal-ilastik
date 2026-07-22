# napari-behavior-classifier

A [napari](https://napari.org) plugin for ilastik-like animal behavior classification from SLEAP pose-tracking data.


> ⚠️ Work in progress ⚠️

## Installation

Requires [uv](https://docs.astral.sh/uv/) and SSH access to the ISTA GitLab.

Install as an isolated tool (recommended — keeps napari and the plugin in their own
environment, exposes the `napari`, `napari-behavior-train`, and `napari-behavior-predict`
commands on your PATH):

```sh
uv tool install "napari-behavior-classifier @ git+ssh://git@git.ista.ac.at/csommer/napari_behavior_classifier.git"
```


## Usage

Launch napari and open the plugin from the menu: **Plugins -> Behavior Classifier**

```sh
napari
```

For batch/headless use:

```sh
napari-behavior-train session.json model.joblib      # pool a saved session into a model
napari-behavior-predict model.joblib file1.h5 file2.h5 ...  # predict, one CSV per file
```

## Updating

```sh
uv tool upgrade napari-behavior-classifier
```

