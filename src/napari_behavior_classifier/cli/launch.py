"""Launch napari with the Behavior Classifier widget already docked.

    anilastik

A single, uniquely-named entry point: it can't be shadowed by a separately
installed `napari` on PATH, and the user doesn't have to hunt for the plugin in
the Plugins menu - it opens exactly as the menu would, via the same npe2
contribution (so viewer injection and setup are identical).
"""

from __future__ import annotations

PLUGIN_NAME = "napari-animal-ilastik"
WIDGET_NAME = "Behavior Classifier"


def main() -> int:
    import napari

    viewer = napari.Viewer()
    viewer.window.add_plugin_dock_widget(PLUGIN_NAME, WIDGET_NAME)
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
