"""Friendly model-role descriptions and drawing-only per-neuron visibility."""
import numpy as np
import torch  # Native DLL order on Windows.
from panda3d.core import Shader
from flybrain.flight3d.branch_activity import BranchActivityTable, VERTEX_SHADER

# These explain project-assigned roles, not a new biological classification.
GROUPS = {
    'odor': ('Smell', 'Receives the simulated flower smell signal.'),
    'vision': ('Vision', "Processes simulated images from the fly's eyes."),
    'optic_flow': ('Visual motion', 'Responds to movement across the eye images.'),
    'motion': ('Body movement sensing', 'Receives simulated body movement signals.'),
    'taste': ('Taste', 'Receives the simulated food-contact signal.'),
    'touch': ('Touch', 'Receives simulated physical contact signals.'),
    'odor_projection': ('Smell relays', 'Passes smell-related activity through the model.'),
    'motor': ('Wing and landing control', 'Modeled control cells for wings and landing.'),
    'descending': ('Movement commands', 'Relays activity toward modeled control cells.'),
    'structural': ('Other relays', 'Cells retained on sensor-to-movement paths.'),
}

# Softer than white, distinct from sensory greens, relay purples and motors.
# This is a dashboard-only color override; the source-bound palette stays intact.
DISPLAY_COLORS = {'structural': (91/255, 143/255, 185/255)}  # #5B8FB9


def group_color(category) -> tuple[float, float, float]:
    """Use the same display color in firing branches and sidebar swatches."""
    return DISPLAY_COLORS.get(category.key, category.rgb)


def display_palette(index) -> np.ndarray:
    """Copy the immutable model palette; recoloring never changes membership."""
    colors = np.array(index.activation_colors, copy=True)
    for category in index.classes:
        colors[list(index.members[category.key])] = group_color(category)
    return colors


class GroupActivityTable(BranchActivityTable):
    """Hide unchecked owners in the GPU without touching their simulated state.

    Zero alpha also discards the fragment, so an invisible branch cannot write
    depth and obscure a visible branch. Existing spike pulses remain unchanged.
    """
    def __init__(self, ids: tuple[str, ...], activation_colors=None) -> None:
        self.visibility = None
        super().__init__(ids, activation_colors)
        self.shader = Shader.make(Shader.SLGLSL, VERTEX_SHADER, '''#version 130
in vec4 neuron_color;
out vec4 fragment_color;
void main() {
    if (neuron_color.a < 0.001) discard;
    fragment_color = neuron_color;
}
''')

    def update(self, frame, bright: bool, mask: tuple[bool, ...] | None = None) -> None:
        self.visibility = mask
        super().update(frame, bright, mask)

    def upload_colors(self, colors: np.ndarray) -> None:
        if self.visibility is not None:
            colors = np.array(colors, copy=True)
            colors[~np.asarray(self.visibility, dtype=bool), 3] = 0
        super().upload_colors(colors)
