"""Engineered RGB input on existing source cells; unchanged assumed LIF graph."""
import numpy as np
from flybrain.flight3d.contracts import FlightObservation
from flybrain.flight3d.neural import FlightBrain


def threat_contrast(rgb: np.ndarray) -> np.ndarray:
    """Cyan contrast from RGB alone; flower, stem, sky and floor are rejected.

    This color-selective front end is engineered, not measured fly physiology.
    Subpixel RGB dilution retains contrast above .05, including small objects.
    """
    values=np.asarray(rgb)
    contrast=np.minimum(values[...,1], values[...,2])-values[...,0]
    return np.where(contrast>.05, contrast, 0.)


class EscapeBrain(FlightBrain):
    """Retain every source ID/edge and original numerical dynamics.

    Vision-only input isolates the escape pathway from unrelated sensory drive.
    The existing ID-ranked 8x6 fields are an explicit assumed retinotopy, not
    biological tuning. Positive RGB image growth strengthens each patch's drive.
    """
    def sensory_rates(self, observation: FlightObservation) -> np.ndarray:
        salience=threat_contrast(observation.eyes)
        growth=np.zeros_like(salience) if self.previous_eyes is None else np.maximum(salience-self.previous_eyes,0.)
        self.previous_eyes=salience.copy()
        rates=np.zeros(len(self.ids))
        for side, eye in (('left',0), ('right',1)):
            group=[int(i) for i in self.roles.get('vision',()) if self.nodes[i]['metadata'].get('side')==side]
            for rank,i in enumerate(group):
                row=(rank%6)*4; column=(rank//6%8)*6
                patch=salience[eye,row:row+4,column:column+6]
                expansion=growth[eye,row:row+4,column:column+6]
                # Peak detects small appearances; occupancy keeps expansion
                # informative after central pixels reach full cyan contrast.
                rates[i]=240*min(1.,.75*float(patch.max())+.25*float(patch.mean())
                                 +.5*float(expansion.max()))
        return rates
