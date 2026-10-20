"""Display-only controls for the anatomical preview and its independent cameras."""
import math
import numpy as np
import torch  # Initialize native DLLs before Panda on Windows.
from direct.gui.DirectGui import DirectButton, DirectScrolledFrame, DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, Point3, Vec3
from .brain_camera import BrainCamera


class BrainControls:
    """UI/navigation mixin; never calculates sensory or neural signals."""
    def _brain_camera(self) -> None:
        if not isinstance(self.brain_motion, BrainCamera):
            self.brain_motion = BrainCamera()
        motion = self.brain_motion
        self.brain_target = motion.centre.copy(); self.distance = motion.distance
        self.brain_camera.setPos(*motion.position)
        self.brain_camera.lookAt(Point3(*motion.centre), Vec3(*motion.up))
        self.brain_camera.node().getLens().setNear(motion.near)

    def preset(self, name: str) -> None:
        self._brain_camera()
        self.brain_motion.preset(name); self._brain_camera()
        self.views_open = False
        if hasattr(self, 'views_menu'):
            for widget in self.views_menu: widget.hide()

    def visible_bounds(self) -> np.ndarray:
        """Bounds use registered displayed geometry, never controller input."""
        rows = []
        for cord in (False, True):
            if cord and not self.vnc: continue
            if self.whole: rows.extend(self._context_bounds[int(cord)])
            rows.extend(self._branch_bounds[int(cord)])
        if not rows: rows = list(self._context_bounds[0])
        points = np.asarray(rows)
        return np.array((points.min(axis=0), points.max(axis=0)))

    def fit_view(self) -> None:
        self.brain_motion.fit(self.visible_bounds(), tuple(self.brain_camera.node().getLens().getFov()))

    def reset_view(self) -> None:
        self.preset('front'); self.fit_view()

    def focus_brain(self) -> None:
        """Legacy B shortcut now means explicit Reset view, including visible cord."""
        self.reset_view()

    def toggle_vnc(self) -> None:
        """Visibility is geometry only: no camera state or target mutation."""
        self.vnc = not self.vnc; self._visibility()
        if hasattr(self, 'vnc_button'):
            self.vnc_button['text'] = 'Show nerve cord: on' if self.vnc else 'Show nerve cord: off'
        if self.selected and self.show_edges: self._draw_connections()
        if self.selected and not self.settings_open:self._show_selection_info()

    def begin_gesture(self, x: float, y: float, pan: bool = False) -> None:
        """Begin only inside a viewport; record pixels and gesture mode."""
        if self._over_view_menu(x,y): return
        panel = self.layout.route(x,y)
        if panel not in ('world','brain'): return
        self.drag = (x,y,x,y)
        self._gesture_panel = panel; self._gesture_pan = pan
        self._gesture_travel = 0.
        self._gesture_size = (self.win.getXSize(),self.win.getYSize())

    def start_drag(self) -> None:
        if self.mouseWatcherNode and self.mouseWatcherNode.hasMouse():
            mouse = self.mouseWatcherNode.getMouse()
            self.begin_gesture(mouse.x,mouse.y,'shift' in self.keys)

    def start_pan(self) -> None:
        self.keys.add('shift'); self.start_drag()

    def move_gesture(self, x: float, y: float) -> None:
        """Convert window-normalized mouse positions to viewport pixel deltas."""
        if not self.drag: return
        if self._gesture_size != (self.win.getXSize(),self.win.getYSize()):
            self.cancel_gesture(); return
        dx = (x-self.drag[2])*self.win.getXSize()/2
        dy = (y-self.drag[3])*self.win.getYSize()/2
        self._gesture_travel += math.hypot(dx,dy)
        panel = self._gesture_panel
        motion = self.brain_motion if panel == 'brain' else self.world_motion
        if self._gesture_pan:
            _,_,bottom,top = self.layout.rectangles[panel]
            camera = self.brain_camera if panel == 'brain' else self.camera
            height = self.win.getYSize()*(top-bottom)/2
            if panel == 'brain': motion.pan_pixels(dx,dy,height,camera.node().getLens().getFov()[1])
            else:
                scale = 2*motion.distance*math.tan(math.radians(camera.node().getLens().getFov()[1])/2)/height
                basis = camera.getQuat()
                # Preserve fly-camera bounds without tying pan to its zoom floor.
                motion.offset += scale*(-dx*np.array(basis.getRight())-dy*np.array(basis.getUp()))
        elif panel == 'brain': motion.rotate_pixels(dx,dy)
        else:
            self.world_azimuth -= dx*.006
            self.world_elevation = float(np.clip(self.world_elevation+dy*.006,-1.4,1.4))
        self.drag = (*self.drag[:2],x,y)

    def stop_drag(self) -> None:
        if not self.drag: return
        if self.mouseWatcherNode and self.mouseWatcherNode.hasMouse():
            mouse = self.mouseWatcherNode.getMouse(); self.move_gesture(mouse.x,mouse.y)
        if self.drag and not self._gesture_pan and self._gesture_travel < 4:
            self.pick(*self.drag[:2])
        self.drag = None; self.brain_motion.release()

    def cancel_gesture(self) -> None:
        """Focus/pointer loss and resize clear gestures without selecting anything."""
        self.drag = None
        if hasattr(self,'keys'): self.keys.clear()
        if isinstance(self.brain_motion,BrainCamera): self.brain_motion.release()

    def _over_view_menu(self, x: float, y: float) -> bool:
        return self.views_open and .20 <= x <= .36 and .44 <= y <= .64

    def zoom_at(self, factor: float, x: float, y: float) -> None:
        """Menu overlays consume wheel input rather than zooming behind them."""
        if self._over_view_menu(x,y): return
        if self.layout.route(x,y) == 'world':
            motion = self.world_motion
            motion.target = float(np.clip(motion.target*factor,motion.minimum,motion.maximum))
        else: super().zoom_at(factor,x,y)

    def windowEvent(self, window) -> None:
        super().windowEvent(window)
        if hasattr(self,'brain_motion') and hasattr(window,'getProperties'):
            properties = window.getProperties()
            self.focus_changed(properties.getForeground() and not properties.getMinimized())

    def focus_changed(self, foreground: bool) -> None:
        """Clear held keys and gestures when the native window loses focus."""
        if not foreground: self.cancel_gesture()

    def toggle_settings(self) -> None:
        self.settings_open = not self.settings_open
        self._restore_inspector()

    def _refresh_activity_choices(self) -> None:
        """Rebuild a small immutable display mask; never reset neural history."""
        if self.selected not in self.activity_index.adjacency:
            self.activity_scope='whole'
        self.activity_mask=self.activity_index.mask(self.enabled_classes,self.activity_scope,self.selected)
        self.scope_count=len(self.activity_index.scope_members(self.activity_scope,self.selected))
        self._activity_token=None
        if self._display_frame is not None:
            self._update_activity(self._display_frame);self._refresh_coverage()
        if hasattr(self,'legend_entries'):self._refresh_legend()
        if self.settings_open and hasattr(self,'info'):
            scroll=self.panels['details'].verticalScroll['value']
            self._restore_inspector();self.panels['details'].verticalScroll['value']=scroll

    def toggle_activity_class(self, key: str) -> None:
        """Toggle one category without changing any other switch or scope."""
        if key not in self.activity_index.members:raise ValueError('unknown display class')
        self.enabled_classes=self.enabled_classes ^ frozenset({key})
        self._refresh_activity_choices()

    def toggle_sensory(self) -> None:
        """Apply the sensory parent on/off/mixed rule."""
        self.enabled_classes=self.activity_index.toggle_sensory(self.enabled_classes)
        self._refresh_activity_choices()

    def toggle_sensory_expansion(self) -> None:
        """Expand/collapse controls without changing their enabled state."""
        self.sensory_expanded=not self.sensory_expanded;self._restore_inspector()

    def all_activity(self) -> None:
        """Enable every class and restore whole-model activity scope."""
        self.enabled_classes=frozenset(self.activity_index.members);self.activity_scope='whole'
        self._refresh_activity_choices()

    def activity_off(self) -> None:
        """Disable colored pulses while retaining scope and all recorded spikes."""
        self.enabled_classes=frozenset();self._refresh_activity_choices()

    def set_activity_scope(self, scope: str) -> None:
        """Selected scope is available only with a modeled selection."""
        if scope not in ('whole','selected'):raise ValueError('unknown activity scope')
        if scope=='selected' and self.selected not in self.activity_index.adjacency:return
        self.activity_scope=scope;self._refresh_activity_choices()

    def _refresh_legend(self) -> None:
        """Dim disabled categories; cyan remains an independent selection cue."""
        for c in self.activity_index.classes:
            swatch,label=self.legend_entries[c.key];enabled=c.key in self.enabled_classes
            swatch['frameColor']=(*tuple(v*(1 if enabled else .38) for v in c.rgb),1)
            label.setText(c.label+' / '+('on' if enabled else 'off'))
            label['fg']=(.86,.83,.94,1) if enabled else (.43,.41,.49,1)

    def _restore_inspector(self) -> None:
        if self.settings_open:
            self._set_info('Display settings / click a class to switch its color on or off\nModel-assigned activity colors; every neuron stays simulated and recorded.')
            rows=[('All activity',self.all_activity),('Activity off',self.activity_off),
                ('Whole model'+(' / selected' if self.activity_scope=='whole' else ''),lambda:self.set_activity_scope('whole')),
                ('Selected circuit'+(' / select modeled neuron' if self.selected not in self.activity_index.adjacency else ' / selected' if self.activity_scope=='selected' else ''),lambda:self.set_activity_scope('selected')),
                ('Sensory: '+self.activity_index.sensory_state(self.enabled_classes)+' ('+str(sum(len(self.activity_index.members[c.key]) for c in self.activity_index.classes[:6]))+')',self.toggle_sensory),
                ('Sensory children: '+('collapse' if self.sensory_expanded else 'expand'),self.toggle_sensory_expansion)]
            for i,c in enumerate(self.activity_index.classes):
                if i<6 and not self.sensory_expanded:continue
                rows.append((c.label+f' ({len(self.activity_index.members[c.key])}) / '+('on' if c.key in self.enabled_classes else 'off'),lambda key=c.key:self.toggle_activity_class(key)))
            rows.extend([('Background anatomy: '+('on' if self.whole else 'off'),self.toggle_mode),
                ('Branch brightness: '+('bright' if self.detail_view else 'soft'),self.toggle_view),
                ('Reset view',self.reset_view),('Close settings',self.toggle_settings)])
            self._buttons(rows)
            self.member_buttons[3]['state']='normal' if self.selected in self.activity_index.adjacency else 'disabled'
            self._layout_activity_settings()
        elif self.selected: self._show_selection_info()
        elif self.inspection: self.list_groups()
        else:
            self._set_info('Selected neuron\nClick branches to inspect identity, model role and source information. Selection leaves the camera in place.')
            self._buttons([('Browse neurons',self.list_groups)])

    def _layout_activity_settings(self) -> None:
        """Put every class switch above the fold, with shortcuts/scope below."""
        aspect=self.getAspectRatio()
        width=(self.layout.rectangles['details'][1]-self.layout.rectangles['details'][0])*aspect-.025
        left=.018;right=left+width/2
        y=float(self.info.getTightBounds()[0].z)-.035
        self.member_buttons[4].setPos(left,0,y)
        self.member_buttons[5].setPos(right,0,y)
        visible=[c for i,c in enumerate(self.activity_index.classes) if i>=6 or self.sensory_expanded]
        for button,c in zip(self.member_buttons[6:6+len(visible)],visible):
            index=self.activity_index.classes.index(c)
            column=left if index<6 else right
            row=index if index<6 else index-6
            button.setPos(column,0,y-.037*(row+1))
            button['relief']=1
            button['text_fg']=(*c.rgb,1) if c.key in self.enabled_classes else (.72,.67,.81,1)
        y-=.037*(max(6 if self.sensory_expanded else 0,4)+1)
        for i,button in enumerate(self.member_buttons[:4]):
            button.setPos(left if i%2==0 else right,0,y-.037*(i//2))
        y-=.074
        for button in self.member_buttons[6+len(visible):]:
            button.setPos(left,0,y);y-=.037
        self.panels['details']['canvasSize']=(0,width,min(-.38,y-.045),0)

    def toggle_views(self) -> None:
        self.views_open = not self.views_open
        for widget in self.views_menu: widget.show() if self.views_open else widget.hide()

    def _ui(self) -> None:
        for button in self.member_buttons: button.destroy()
        for node in self.ui_nodes: node.destroy() if hasattr(node,'destroy') else node.removeNode()
        self.ui_nodes = []; self.member_buttons = []
        aspect = self.getAspectRatio(); self.ui_size = (self.win.getXSize(),self.win.getYSize())
        def text(value,x,y,size=.025,color=(.86,.83,.94,1)):
            node = OnscreenText(text=value,pos=(x*aspect,y),align=TextNode.ALeft,
                scale=min(size,aspect*.025),fg=color,mayChange=True)
            self.ui_nodes.append(node); return node
        def button(value,x,y,fn):
            node = DirectButton(text=value,pos=(x*aspect,0,y),scale=min(.023,aspect*.015),command=fn,
                frameColor=(.10,.065,.16,1),text_fg=(.91,.86,.98,1),pad=(.6,.4),relief=0)
            self.ui_nodes.append(node); return node
        self.title = text('F L Y B R A I N   /   FLIGHT LAB',-.95,.91,.043)
        text('BANC v888 | 589 simulated cells | sourced graph, modeled physiology',-.95,.85,.023)
        self.activity_label = text('OBSERVATIONAL activity | teacher supplies commands',.04,.85,.023)
        text('01   FLOWER LANDING',-.95,.77,.03)
        self.view_title = text('02   FLY BRAIN / SOURCE ANATOMY',.04,.77,.03)
        button('Follow / F',-.83,.68,self.toggle_follow); button('Reset camera',-.55,.68,self.reset_world)
        button('Fit view',.105,.68,self.fit_view)
        self.views_button = None; self.views_menu = []
        if self.ui_size[0] < 1200:
            self.views_button = button('Views',.28,.68,self.toggle_views)
            for index,name in enumerate(('Front','Side','Top')):
                widget = button(name,.28,.61-index*.055,lambda p=name.lower():self.preset(p))
                widget.setBin('fixed',100); widget.hide(); self.views_menu.append(widget)
        else:
            for name,x in (('Front',.24),('Side',.335),('Top',.425)):
                button(name,x,.68,lambda p=name.lower():self.preset(p))
        self.views_open = False
        self.vnc_button = button('Show nerve cord: '+('on' if self.vnc else 'off'),.62 if self.views_button is None else .52,.68,self.toggle_vnc)
        button('Display settings',.855,.68,self.toggle_settings)
        self.hud = text('',-.95,-.415,.024)
        self.coverage = text('',.04,-.405,.019)
        self.panels = {}
        for name in ('help','details'):
            left,right,bottom,top = self.layout.rectangles[name]
            frame = DirectScrolledFrame(frameSize=(left*aspect,right*aspect,bottom,top),
                canvasSize=(0,(right-left)*aspect,-1,0),scrollBarWidth=.025,
                frameColor=(.025,.018,.043,1),borderWidth=(0,0))
            self.ui_nodes.append(frame); self.panels[name] = frame
        for name in ('info','help'):
            panel = self.panels['details' if name == 'info' else 'help']
            node = OnscreenText(parent=panel.getCanvas(),text='',pos=(.018,-.035),align=TextNode.ALeft,
                scale=.025,fg=(.86,.83,.94,1) if name == 'info' else (.64,.60,.74,1),
                mayChange=True,wordwrap=max(20,aspect*34))
            self.ui_nodes.append(node); setattr(self,name,node)
        self.legend_entries={}
        panel=self.panels['help'];width=(self.layout.rectangles['help'][1]-self.layout.rectangles['help'][0])*aspect-.025
        title=OnscreenText(parent=panel.getCanvas(),text='Activity colors / model assignments',pos=(.018,-.035),align=TextNode.ALeft,scale=.023,fg=(.86,.83,.94,1))
        self.ui_nodes.append(title)
        for i,c in enumerate(self.activity_index.classes):
            x=.018+(i//5)*width/2;y=-.073-(i%5)*.031
            swatch=DirectFrame(parent=panel.getCanvas(),pos=(x,0,y+.006),frameSize=(0,.016,-.008,.008),relief=1)
            label=OnscreenText(parent=panel.getCanvas(),text='',pos=(x+.028,y),align=TextNode.ALeft,scale=.024,mayChange=True)
            self.ui_nodes.extend((swatch,label));self.legend_entries[c.key]=(swatch,label)
        self._refresh_legend();self.help.setPos(.018,-.255)
        self.help.setText('Blue/violet: idle anatomy. Cyan: selected branches.\n'
            'Off hides class glow; it does not mean neuronal silence.\n'
            'After each spike, color fades smoothly to idle over 600 ms. Pause freezes the fade.\n\n'
            'Drag: rotate. Shift-drag: pan. Wheel over viewport: zoom.\n'
            'Wheel over these panels: scroll. 1 / 2 / 3: front / side / top.\n'
            'Fit view frames visible anatomy. Presets keep pan and zoom.\n'
            'Tab: Background anatomy. V: nerve cord. B: Reset view.\n'
            'F: follow fly. R: new trial. Backspace: replay. Space: pause. Esc: exit.\n'
            'Manual: Up/Down power; W/S pitch; A/D roll; Q/E yaw; L legs.\n\n'
            'Background anatomy: faint registered surface/reference points.\n'
            'Background anatomy does not identify trained cells.\n'
            'Whole-branch light indicates activity; signal position and conduction speed are not measured.\n'
            'Teacher activity is observational. Brightness does not establish importance.')
        self.panels['help']['canvasSize'] = (0,.9*aspect,min(-.4,float(self.help.getTightBounds()[0].z)-.04),0)
        for value,x,fn in (('Pause / Space',-.79,self.toggle_pause),('New trial / R',-.42,self.new_trial),
            ('Replay / Backspace',.04,self.replay),('Neurons',.49,self.list_groups),('Exit / Esc',.82,self.userExit)):
            button(value,x,-.92,fn)
        self._restore_inspector(); self._coverage_text = None; self._refresh_coverage()
