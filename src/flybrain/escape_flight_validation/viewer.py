"""New prototype presentation; frozen physiology and old launchers preserved."""
from panda3d.core import WindowProperties
from flybrain.escape_recovery.viewer import RecoveryViewer
from .acceptance import LABEL


class PrototypeViewer(RecoveryViewer):
    """Display the accepted directional limitation beside validated flight."""
    def __init__(self,seed,candidate,**kwargs):
        super().__init__(seed,candidate,verified=True,**kwargs)
        if not self.offscreen:
            props=WindowProperties();props.setTitle('FlyBrain | '+LABEL);self.win.requestProperties(props)

    def draw_frame(self,frame):
        super().draw_frame(frame)
        self.title.setText('F L Y B R A I N   /   PHYSICAL ESCAPE PROTOTYPE')
        self.pathway_label.setText(LABEL+'\nKnown directional failures: 1100018, 1100021 (86.25 degrees)\n'+self.pathway_label.getText())
