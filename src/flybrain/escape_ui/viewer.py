"""Modern dark dashboard around the original source-bound verified simulation."""
from pathlib import Path
import numpy as np
import torch  # Windows native DLL initialization precedes Panda3D.
from direct.gui.DirectGui import DirectScrolledFrame
from panda3d.core import Filename, TextNode, WindowProperties

from flybrain.escape_edge.viewer import EdgeViewer
from flybrain.escape_edge.session import EdgeSession
from flybrain.escape_recovery.config import CandidateConfig
from .filters import GroupFilters
from .groups import GROUPS, group_color
from .inspector import DashboardInspector
from .layout import DashboardLayout, Rect
from .widgets import Widgets, INK, MUTED, PANEL, BUTTON, ACCENT

HELP = '''Explore the simulation

Spawn a flower
Watch teacher-guided flight toward a flower. The brain view records activity.

Spawn a predator
Neural activity chooses a turn. An engineered controller flies to the edge and hovers.

Explore the brain
Tick groups to show their branches. Untick to hide them. Every neuron still runs and records activity.
Click a branch to inspect its role and source type. Group colors light up when neurons fire.

Move the view
Drag to rotate. Shift-drag to pan. Scroll over a view to zoom. Fit frames visible anatomy.

Move the predator
W/S forward and back; A/D left and right; Q/E down and up.

Simulation controls
Space: pause or resume. R: new trial. Backspace: rebuild the recorded interaction and pause. Esc: exit.

Display controls
F: follow fly. B: reset brain view. V: nerve cord. Tab: faint brain outline.

What you are seeing
The checkbox groups are model roles. Source cell types are shown in Inspect. Brightness shows recorded activity, not biological importance or learning.
The current demo uses Panda3D and frozen PyTorch forward calculations. Earlier 2D work uses Pygame and has separate learned readouts.
'''


class DemoViewer(DashboardInspector, GroupFilters, EdgeViewer):
    """Native checkboxes filter owned geometry, never neural computation.

    The original session, worker barriers and action log remain unchanged.
    Pixels define the GUI; camera/input rectangles derive from the same layout.
    """
    def __init__(self, seed: int, candidate: CandidateConfig,
                 verified: bool = False, **kwargs) -> None:
        self.sidebar_tab = 'groups'
        self.focused_group = 'vision'
        self.technical_open = False
        self.browser_page = 0
        self.group_bounds = {}
        self._gui_hits = []
        super().__init__(seed, candidate, verified=verified, **kwargs)
        if not self.offscreen:
            props = WindowProperties()
            props.setTitle('FlyBrain | Explore the brain')
            self.win.requestProperties(props)

    def _load_ui_font(self) -> None:
        if hasattr(self, 'ui_font'):
            return
        # Use the readable native Windows font without copying/distributing it.
        # Panda3D's built-in font remains the portable fallback.
        path = Path('C:/Windows/Fonts/segoeui.ttf')
        self.ui_font = None
        if path.exists():
            try:
                self.ui_font = self.loader.loadFont(str(Filename.fromOsSpecific(str(path))))
            except OSError:
                pass  # A missing/unreadable native font must not prevent launch.

    def _scene_button(self, text, x, y, width, callback, height=34, size=14):
        node = self.widgets.button(text, x, y, width, callback, height=height, size=size)
        self._gui_hits.append((node, Rect(x, y-height/2, x+width, y+height/2)))
        return node

    def _ui(self) -> None:
        """Rebuild layout on resize while retaining filters, tab and inspection."""
        scroll = {name: panel.verticalScroll['value']
                  for name, panel in getattr(self, 'panels', {}).items()}
        for button in self.member_buttons:
            button.destroy()
        for node in self.ui_nodes:
            node.destroy() if hasattr(node, 'destroy') else node.removeNode()
        self.ui_nodes, self.member_buttons, self._gui_hits = [], [], []
        width, height = self.win.getXSize(), self.win.getYSize()
        self.ui_size = (width, height)
        self.layout = DashboardLayout(width, height)
        self._load_ui_font()
        self.widgets = Widgets(self.pixel2d, self.ui_nodes, self.ui_font)
        w = self.widgets
        self.views_open, self.views_menu, self.views_button = False, [], None
        self.setBackgroundColor(.02, .027, .043)
        self.brain_camera.node().getDisplayRegion(0).setClearColor((.02, .027, .043, 1))

        w.panel(20, 22, 6, 31, ACCENT, radius=3)
        self.title = w.label('FlyBrain', 38, 47, 30)
        w.label('Explore the fly. Understand the brain.', 22, 71, 16, MUTED)
        w.panel(width-270, 24, 250, 34, (.05, .12, .16, 1))
        w.label('589 modeled neurons', width-254, 47, 16, ACCENT)
        self.activity_label = w.label('', width-425, 72, 14, MUTED)

        x = 20
        self.action_buttons = {}
        for key, text, button_width, command, color in (
            ('flower', 'Spawn flower', 148, lambda: self.spawn('flower'), (.10, .28, .23, 1)),
            ('predator', 'Spawn predator', 160, lambda: self.spawn('predator'), (.06, .29, .37, 1)),
            ('pause', 'Pause', 118, self.toggle_pause, BUTTON),
            ('reset', 'New trial', 120, self.new_trial, BUTTON),
            ('replay', 'Replay', 100, self.replay, BUTTON)):
            self.action_buttons[key] = w.button(text, x, 106, button_width, command, height=46, color=color)
            x += button_width+10
        self.spawn_flower_button = self.action_buttons['flower']
        self.spawn_predator_button = self.action_buttons['predator']
        self.pause_button = self.action_buttons['pause']
        self.new_trial_button = self.action_buttons['reset']
        self.replay_button = self.action_buttons['replay']
        w.button('Exit', width-84, 106, 64, self.userExit, height=46)

        for name, card in self.layout.cards.items():
            # GUI backgrounds cover only chrome, leaving camera pixels exposed.
            w.panel(card.left, card.top, card.width, 48)
            w.panel(card.left, card.bottom-46, card.width, 46)
            w.panel(card.left, card.top+48, 1, card.height-94, (.10, .14, .19, 1), radius=0)
            w.panel(card.right-1, card.top+48, 1, card.height-94, (.10, .14, .19, 1), radius=0)
        world, brain = self.layout.cards['world'], self.layout.cards['brain']
        w.label('World', world.left+16, world.top+30, 20)
        self.follow_button = self._scene_button('Follow', world.right-180, world.top+24, 70, self.toggle_follow)
        self._scene_button('Reset view', world.right-100, world.top+24, 84, self.reset_world)
        self.view_title = w.label('Brain', brain.left+16, brain.top+30, 20)
        self._scene_button('Fit visible', brain.right-110, brain.top+24, 94, self.fit_view)
        self.world_caption = w.label('Spawn a flower or predator', world.left+14, world.bottom-17, 16, MUTED)
        preset_width = (brain.width-44)/3
        for index, name in enumerate(('Front', 'Side', 'Top')):
            self._scene_button(name, brain.left+16+index*(preset_width+6), brain.bottom-23,
                               preset_width, lambda name=name.lower(): self.preset(name))
        self.hud = w.label('', 22, height-18, 16, MUTED)

        side = self.layout.sidebar
        w.panel(side.left, side.top, side.width, side.height)
        self.tab_buttons = {}
        for index, tab in enumerate(('groups', 'inspect', 'help')):
            self.tab_buttons[tab] = w.button(tab.title(), side.left+12+index*100,
                side.top+28, 96, lambda tab=tab: self.show_tab(tab), height=36, size=16)
        self.group_root = self.pixel2d.attachNewNode('groups-page')
        self.ui_nodes.append(self.group_root)
        gp = self.group_root
        w.label('MODEL ROLE GROUPS', side.left+16, 208, 14, MUTED, gp)
        bulk_width = (side.width-48)/3
        for i, (text, command) in enumerate((('Show all', self.show_all_groups),
                ('Hide all', self.hide_all_groups), ('Only this', lambda: self.show_only_group(self.focused_group)))):
            w.button(text, side.left+16+i*(bulk_width+8), 229, bulk_width,
                     command, height=28, size=14, parent=gp)
        self.group_checks, self.group_counts = {}, {}
        for category, (row_x, row_y) in zip(self.activity_index.classes, self.layout.group_rows):
            key = category.key
            self.group_checks[key] = w.check(GROUPS[key][0], row_x, row_y, side.width-32,
                key in self.enabled_classes, lambda value, key=key: self.set_group_visible(key, bool(value)), parent=gp)
            w.panel(row_x+27, row_y-8, 4, 16, (*group_color(category), 1), gp, radius=2)
            self.group_counts[key] = w.label(str(len(self.activity_index.members[key])),
                side.right-22, row_y+5, 16, MUTED, gp, align=TextNode.ARight)
        self.outline_check = w.check('Faint brain outline', side.left+16, self.layout.outline_y,
                                    side.width-32, self.whole, self.set_outline, parent=gp)
        self.cord_check = w.check('Include nerve cord', side.left+16, self.layout.cord_y,
                                 side.width-32, self.vnc, self.set_cord, parent=gp)
        self.group_name = w.label('', side.left+16, self.layout.description_top+16, 18, ACCENT, gp)
        self.group_description = w.label('', side.left+16, self.layout.description_top+38,
                                         14, MUTED, gp, width=side.width-32)

        self.panels = {}
        for name in ('details', 'help'):
            panel = DirectScrolledFrame(parent=self.pixel2d,
                pos=(side.left+12, 0, -side.top-58),
                frameSize=(0, side.width-24, -side.height+74, 0),
                canvasSize=(0, side.width-48, -side.height, 0),
                frameColor=PANEL, scrollBarWidth=12, borderWidth=(0, 0),
                verticalScroll_frameColor=PANEL, horizontalScroll_frameColor=PANEL)
            self.ui_nodes.append(panel)
            self.panels[name] = panel
        self.info = w.label('', 8, 26, 16, parent=self.panels['details'].getCanvas(), width=side.width-64)
        self.help = w.label(HELP, 8, 26, 16, parent=self.panels['help'].getCanvas(), width=side.width-64)
        self.panels['help']['canvasSize'] = (0, side.width-48, float(self.help.getTightBounds()[0].z)-30, 0)
        # These legacy labels are required by inherited drawing, but detailed
        # measurements belong in our opt-in inspector rather than the dashboard.
        self.pathway_label = w.label('', 0, 0)
        self.pathway_label.hide()
        self.coverage = w.label('', 0, 0)
        self.coverage.hide()
        self.legend_entries = {}
        self._coverage_text = None
        view = self.layout.viewports['brain']
        self.empty_label = w.label('No neuron groups selected', (view.left+view.right)/2,
            (view.top+view.bottom)/2-20, 18, parent=None, align=TextNode.ACenter)
        self.empty_button = self._scene_button('Show all groups', (view.left+view.right)/2-86,
            (view.top+view.bottom)/2+20, 172, self.show_all_groups, height=44, size=16)
        self._restore_inspector()
        self.show_tab(self.sidebar_tab)
        self._sync_group_controls()
        self._update_pause_button()
        for name, value in scroll.items():
            self.panels[name].verticalScroll['value'] = value

    def _resize(self) -> None:
        super()._resize()
        # pixel2d is independent of the inherited aspect2d navigation/UI.
        self.pixel2d.setScale(2/self.ui_size[0], 1, 2/self.ui_size[1])

    def show_tab(self, tab: str) -> None:
        """Switch navigation without rebuilding the scene or simulation."""
        if tab not in ('groups', 'inspect', 'help'):
            raise ValueError('unknown dashboard tab')
        self.sidebar_tab = tab
        if not hasattr(self, 'group_root'):
            return
        self.group_root.show() if tab == 'groups' else self.group_root.hide()
        self.panels['details'].show() if tab == 'inspect' else self.panels['details'].hide()
        self.panels['help'].show() if tab == 'help' else self.panels['help'].hide()
        for name, button in self.tab_buttons.items():
            button['text_fg'] = ACCENT if name == tab else MUTED
        if tab == 'inspect':
            self._restore_inspector()

    def toggle_settings(self) -> None:
        self.show_tab('groups')

    def _refresh_legend(self) -> None:
        self._sync_group_controls()

    def _sync_group_controls(self) -> None:
        if not hasattr(self, 'group_checks'):
            return
        for key, checkbox in self.group_checks.items():
            enabled = key in self.enabled_classes
            checkbox['indicatorValue'] = int(enabled)
            checkbox['text_fg'] = INK if enabled else MUTED
        self.outline_check['indicatorValue'] = int(self.whole)
        self.cord_check['indicatorValue'] = int(self.vnc)
        name, description = GROUPS[self.focused_group]
        self.group_name.setText(name)
        self.group_description.setText(description)
        any_group = any(self.activity_mask)
        self.empty_label.hide() if any_group else self.empty_label.show()
        self.empty_button.hide() if any_group else self.empty_button.show()

    def _update_pause_button(self) -> None:
        if hasattr(self, 'pause_button'):
            self.pause_button['text'] = 'Resume' if self.paused else 'Pause'

    def toggle_pause(self) -> None:
        super().toggle_pause()
        self._update_pause_button()

    def _over_gui(self, x: float, y: float) -> bool:
        px, py = (x+1)*self.ui_size[0]/2, (1-y)*self.ui_size[1]/2
        return any(not node.isHidden() and rect.contains(px, py) for node, rect in self._gui_hits)

    def begin_gesture(self, x: float, y: float, pan: bool = False) -> None:
        if not self._over_gui(x, y):
            super().begin_gesture(x, y, pan)

    def zoom_at(self, factor: float, x: float, y: float) -> None:
        if self._over_gui(x, y):
            return
        if self.layout.route(x, y) in ('details', 'help'):
            if self.sidebar_tab != 'groups':
                name = 'details' if self.sidebar_tab == 'inspect' else 'help'
                bar = self.panels[name].verticalScroll
                bar['value'] = float(np.clip(bar['value']+(.09 if factor > 1 else -.09), 0, 1))
            return
        super().zoom_at(factor, x, y)

    def draw_frame(self, frame) -> None:
        """Draw one published snapshot; no RNG use or neural/physics stepping."""
        super().draw_frame(frame)
        if not isinstance(getattr(self, 'session', None), EdgeSession):
            return
        s = frame.snapshot
        self.title.setText('FlyBrain')
        state = {'takeoff': 'Taking off', 'turning': 'Turning away', 'fleeing': 'Flying to edge',
                 'edge_braking': 'Slowing near edge', 'edge_hover': 'Hovering at edge'}.get(
                     s.behavior_state, s.behavior_state.replace('_', ' ').capitalize())
        obj = 'No object' if s.object_kind == 'none' else s.object_kind.title()
        self.hud.setText(f'{"Paused" if self.paused else "Running"}  /  {state}  /  {obj}'
                         + ('  /  Collision' if s.crashed else ''))
        self.activity_label.setText('Recorded spikes / display filters only')
        self.world_caption.setText('Predator: WASD + Q/E' if s.object_kind == 'predator'
            else 'Teacher-guided flower flight' if s.object_kind == 'flower' else 'Spawn a flower or predator')
        self.follow_button['text'] = 'Following' if self.follow else 'Follow'
        self._sync_selection_visibility()
        self._update_pause_button()
        self._sync_group_controls()
