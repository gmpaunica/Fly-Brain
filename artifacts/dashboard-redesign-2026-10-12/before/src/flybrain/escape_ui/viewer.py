"""Panda3D presentation only; simulation and recorded science stay unchanged."""
import torch  # Initialize the Windows neural DLLs before importing Panda3D.
from direct.gui import DirectGuiGlobals as DGG
from direct.gui.DirectGui import DirectButton, DirectFrame, DirectScrolledFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, WindowProperties

from flybrain.encounters3d.diagnostic_panel import pathway_text
from flybrain.escape_edge.session import EdgeSession
from flybrain.escape_edge.viewer import EdgeViewer
from flybrain.escape_recovery.config import CandidateConfig

INK = (.91, .94, .98, 1)
MUTED = (.64, .71, .80, 1)
SURFACE = (.035, .049, .074, 1)
BUTTON = (.09, .13, .19, 1)


class DemoViewer(EdgeViewer):
    """Larger controls around the retained world and anatomical brain views.

    UI callbacks reuse the existing worker barriers and action log. Display
    updates consume an immutable frame; expanding diagnostics changes no state
    in the world, controller, PyTorch tensors or simulation random generators.
    """

    def __init__(self, seed: int, candidate: CandidateConfig,
                 verified: bool = False, **kwargs) -> None:
        self.diagnostics_open = False
        super().__init__(seed, candidate, verified=verified, **kwargs)
        if not self.offscreen:
            properties = WindowProperties()
            properties.setTitle('FlyBrain | Interactive demo')
            self.win.requestProperties(properties)

    def _font(self, pixels: float) -> float:
        return pixels * 2 / self.ui_size[1]

    def _text(self, value: str, x: float, y: float, pixels: float = 14,
              color: tuple = INK, parent=None, wrap: float | None = None):
        options = {} if wrap is None else {'wordwrap': wrap}
        node = OnscreenText(parent=parent or self.aspect2d, text=value,
            pos=(x, y), scale=self._font(pixels), align=TextNode.ALeft,
            fg=color, mayChange=True, **options)
        self.ui_nodes.append(node)
        return node

    def _button(self, value: str, x: float, y: float, width: float,
                command, color: tuple = BUTTON) -> DirectButton:
        aspect = self.getAspectRatio()
        half_height = 21 / self.ui_size[1] * 2
        node = DirectButton(parent=self.aspect2d, text=value,
            pos=(x * aspect, 0, y), frameSize=(-width * aspect / 2,
            width * aspect / 2, -half_height, half_height),
            text_scale=self._font(14), text_pos=(0, -self._font(5)),
            text_fg=INK, relief=DGG.FLAT, command=command,
            frameColor=(color, tuple(min(1, v * 1.3) for v in color[:3]) + (1,),
                        tuple(v * .8 for v in color[:3]) + (1,), BUTTON))
        self.ui_nodes.append(node)
        return node

    def _ui(self) -> None:
        """Rebuild presentation widgets while retaining inspection preferences."""
        scrolls = {name: panel.verticalScroll['value']
                   for name, panel in getattr(self, 'panels', {}).items()}
        for button in self.member_buttons:
            button.destroy()
        for node in self.ui_nodes:
            node.destroy() if hasattr(node, 'destroy') else node.removeNode()
        self.ui_nodes = []
        self.member_buttons = []
        self.ui_size = (self.win.getXSize(), self.win.getYSize())
        aspect = self.getAspectRatio()
        self.views_open = False
        self.views_menu = []
        self.views_button = None
        self.title = self._text('FlyBrain', -.95 * aspect, .91, 25)
        self._text('Interactive flight + spiking brain', -.95 * aspect, .85, 14, MUTED)
        self._text('BANC v888  /  589 modeled cells', .04 * aspect, .91, 14, MUTED)
        self.activity_label = self._text('Actual recorded spikes', .04 * aspect, .85, 13)
        self._text('WORLD', -.95 * aspect, .77, 13, MUTED)
        self.view_title = self._text('BRAIN / SOURCE ANATOMY', .04 * aspect, .77, 13, MUTED)
        self.spawn_flower_button = self._button('Spawn flower', -.735, .68, .43,
            lambda: self.spawn('flower'), (.09, .27, .19, 1))
        self.spawn_predator_button = self._button('Spawn predator', -.265, .68, .43,
            lambda: self.spawn('predator'), (.08, .24, .31, 1))
        self._button('Fit', .095, .68, .12, self.fit_view)
        for name, x in (('Front', .25), ('Side', .405), ('Top', .56)):
            self._button(name, x, .68, .14, lambda p=name.lower(): self.preset(p))
        self.settings_button = self._button('Display settings', .81, .68, .31,
                                            self.toggle_settings)
        self.hud = self._text('', -.95 * aspect, -.418, 13)
        self.coverage = self._text('', .04 * aspect, -.418, 11, MUTED)
        self.panels = {}
        for name in ('help', 'details'):
            left, right, bottom, top = self.layout.rectangles[name]
            frame = DirectScrolledFrame(parent=self.aspect2d,
                frameSize=(left * aspect, right * aspect, bottom, top),
                canvasSize=(0, (right-left) * aspect, -1, 0),
                scrollBarWidth=self._font(12), frameColor=SURFACE,
                borderWidth=(0, 0), horizontalScroll_frameColor=SURFACE,
                verticalScroll_frameColor=SURFACE)
            self.ui_nodes.append(frame)
            self.panels[name] = frame
        # Keep the inherited inspector and activity-category controls available.
        wrap = (.89 * aspect) / self._font(13)
        self.info = self._text('', .025, -.045, 13,
            parent=self.panels['details'].getCanvas(), wrap=wrap)
        canvas = self.panels['help'].getCanvas()
        self.help = self._text('Spawn a flower to watch teacher-guided landing.\n'
            'Spawn a predator to trigger neural turn + engineered edge flight.\n'
            'Drag to orbit; Shift-drag to pan; wheel to zoom.\n'
            'Move predator: W/S forward/back, A/D left/right, Q/E down/up.',
            .025, -.045, 13, MUTED, canvas, wrap)
        self.pathway_label = self._text('', .025, -.045, 13, INK, canvas, wrap)
        self.legend_entries = {}
        self.legend_title = self._text('ACTIVITY COLORS / MODEL ASSIGNMENTS',
                                      .025, -.25, 12, MUTED, canvas)
        for i, category in enumerate(self.activity_index.classes):
            x = .025 + (i // 5) * .45 * aspect
            y = -.29 - (i % 5) * self._font(20)
            swatch = DirectFrame(parent=canvas, pos=(x, 0, y + .008),
                frameSize=(0, .018, -.009, .009), relief=DGG.FLAT)
            label = self._text('', x + .028, y, 12, parent=canvas)
            self.ui_nodes.append(swatch)
            self.legend_entries[category.key] = (swatch, label)
        self._refresh_legend()
        self.pause_button = self._button('Pause / Space', -.80, -.92, .30, self.toggle_pause)
        self.new_trial_button = self._button('New trial / R', -.48, -.92, .30, self.new_trial)
        self.replay_button = self._button('Replay / Backspace', -.16, -.92, .30, self.replay)
        self.neurons_button = self._button('Neurons', .18, -.92, .28, self.list_groups)
        self.diagnostics_button = self._button('Diagnostics', .49, -.92, .30,
                                                self.toggle_diagnostics)
        self._button('Exit / Esc', .815, -.92, .29, self.userExit)
        self.action_buttons = dict(flower=self.spawn_flower_button,
            predator=self.spawn_predator_button, pause=self.pause_button,
            reset=self.new_trial_button, replay=self.replay_button,
            diagnostics=self.diagnostics_button)
        self._restore_inspector()
        self._coverage_text = None
        self._refresh_coverage()
        self._layout_help()
        self._update_pause_button()
        for name, value in scrolls.items():
            self.panels[name].verticalScroll['value'] = value

    def _layout_help(self) -> None:
        """Place legend beneath whichever text block is expanded."""
        self.pathway_label.show() if self.diagnostics_open else self.pathway_label.hide()
        self.help.hide() if self.diagnostics_open else self.help.show()
        node = self.pathway_label if self.diagnostics_open else self.help
        bounds = node.getTightBounds()
        bottom = float(bounds[0].z) if bounds else -.05
        title_y = bottom - self._font(30)
        self.legend_title.setPos(.025, title_y)
        for i, (swatch, label) in enumerate(self.legend_entries.values()):
            y = title_y - self._font(24 + (i % 5) * 20)
            swatch.setZ(y + .008)
            label.setPos(.025 + (i // 5) * .45 * self.getAspectRatio() + .028, y)
        self.panels['help']['canvasSize'] = (0, .90 * self.getAspectRatio(),
            min(-.38, title_y - self._font(130)), 0)
        self.diagnostics_button['text'] = 'Hide diagnostics' if self.diagnostics_open else 'Diagnostics'

    def toggle_diagnostics(self) -> None:
        """Expand measured pathway details without resetting display history."""
        self.diagnostics_open = not self.diagnostics_open
        self._layout_help()
        self.panels['help'].verticalScroll['value'] = 0

    def _update_pause_button(self) -> None:
        if hasattr(self, 'pause_button'):
            self.pause_button['text'] = 'Resume / Space' if self.paused else 'Pause / Space'

    def _buttons(self, rows) -> None:
        """Keep inherited inspector actions, with readable text and row spacing."""
        super()._buttons(rows)
        for button in self.member_buttons:
            button.setScale(self._font(12))
            button['text_wordwrap'] = .86 * self.getAspectRatio() / self._font(12)
            button['relief'] = DGG.FLAT
        self._place_inspector_rows()

    def _place_inspector_rows(self) -> None:
        y = float(self.info.getTightBounds()[0].z) - self._font(24)
        for button in self.member_buttons:
            button.setZ(y)
            y -= self._font(38)
        self.panels['details']['canvasSize'] = (0, .90 * self.getAspectRatio(),
                                                min(-.38, y - self._font(20)), 0)

    def _layout_activity_settings(self) -> None:
        """Retain the class columns, stretching their inherited compact spacing."""
        super()._layout_activity_settings()
        anchor = float(self.info.getTightBounds()[0].z) - .035
        factor = self._font(38) / .037
        for button in self.member_buttons:
            button.setZ(anchor + (button.getZ() - anchor) * factor)
            button['text_wordwrap'] = .41 * self.getAspectRatio() / self._font(12)
        bottom = min(button.getZ() for button in self.member_buttons)
        self.panels['details']['canvasSize'] = (0, .90 * self.getAspectRatio(),
                                                min(-.38, bottom - self._font(48)), 0)

    def _refresh_selected_info(self, frame) -> None:
        previous = self._selection_info_token
        super()._refresh_selected_info(frame)
        if previous != self._selection_info_token and self.selected and not self.settings_open:
            self._place_inspector_rows()

    def toggle_pause(self) -> None:
        """Use the existing pause barrier and refresh the button immediately."""
        super().toggle_pause()
        self._update_pause_button()

    def draw_frame(self, frame) -> None:
        """Present published state with concise labels and optional diagnostics."""
        super().draw_frame(frame)
        if not isinstance(getattr(self, 'session', None), EdgeSession):
            return
        snapshot = frame.snapshot
        self.title.setText('FlyBrain')
        state = snapshot.behavior_state.replace('_', ' ').title()
        self.hud.setText(f'{"PAUSED" if self.paused else "RUNNING"}  /  {state}  /  '
            f'{snapshot.object_kind.title()}  /  {snapshot.time:.2f}s'
            + ('  /  CRASH' if snapshot.crashed else ''))
        if snapshot.object_kind == 'flower':
            label = 'Flower: teacher-guided flight; brain activity observed'
        elif snapshot.object_kind == 'predator':
            label = 'Predator: neural turn + engineered flight'
        else:
            label = 'Idle: engineered flight; actual spikes displayed'
        self.activity_label.setText(label)
        verified = 'Edge protocol verified: 48 intact cases' if self.edge_verified else 'Development preview: edge flight unverified'
        self.pathway_label.setText(f'{verified}\nTrial seed {snapshot.seed}\n'
            'Historical direction: 94/96; two known failures.\n'
            'PyTorch forward simulation; no online learning.\n'
            f'Escape: {snapshot.escape_state} / turn {snapshot.selected_side or "pending"}\n\n'
            + pathway_text(frame, self.manifest).replace(
                'No escape decoder; spikes do not command retreat.',
                'Paired steering spikes select the turn; edge flight is authored.'))
        self._update_pause_button()
        self._layout_help()

