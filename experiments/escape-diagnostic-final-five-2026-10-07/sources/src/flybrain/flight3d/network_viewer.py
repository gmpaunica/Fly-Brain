"""Readable grouped source network, with the detailed anatomical view retained."""
import math
import numpy as np
import torch  # Load native Torch DLLs before Panda3D on Windows.
from direct.gui.DirectGui import DirectButton, DirectScrolledFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, ClockObject, Point3, Vec3, PNMImage
from .viewer import FlightViewer as AnatomyViewer, geometry
from .network import group_network
from .navigation import PanelLayout, CameraMotion


class FlightViewer(AnatomyViewer):
    """Connection geometry is a diagram, never reconstructed axon geometry."""
    def __init__(self,*args,**kwargs):
        self.layout=PanelLayout(); self.network_view=True; self.selected_group=None
        self.brain_motion=CameraMotion(.96,.65,3.)
        self.world_motion=CameraMotion(1.05,.08,2.)
        self.ui_nodes=[]; self.member_buttons=[]; self.info_text='Click a network group to inspect its source members.'
        self.inspection=None
        super().__init__(*args,**kwargs)
        self.accept('n',self.toggle_view)
        self.accept('shift',self.keys.add,['shift']);self.accept('shift-up',self.keys.discard,['shift'])
        self.accept('shift-mouse1',self.start_pan);self.accept('shift-mouse1-up',self.stop_drag)
        self.focus_brain(); self._visibility()
        self._resize()

    def _anatomy(self):
        super()._anatomy()
        self.grouped=group_network(self.manifest)
        self.network_root=self.brain_scene.attachNewNode('grouped-connection-diagram')
        self.bundle_nodes=[]; self.glow_nodes=[]; self.group_nodes=[]
        for i,bundle in enumerate(self.grouped.bundles):
            a=self.grouped.groups[bundle.pre].centroid; b=self.grouped.groups[bundle.post].centroid
            if a is None or b is None:
                self.bundle_nodes.append(None);self.glow_nodes.append(None);continue
            a=np.array(a);b=np.array(b)
            if bundle.pre==bundle.post:
                # Small closed connection loop represents retained intragroup edges.
                points=np.array([a+.007*np.array([math.cos(t),0,math.sin(t)]) for t in np.linspace(0,math.tau,13)])
            else: points=np.array([a,b])
            lines=np.stack((points[:-1],points[1:]),axis=1)
            node=geometry(f'bundle-{i}',lines,kind='lines',color=(.34,.17,.49,.42))
            node.setRenderModeThickness(1.5);node.reparentTo(self.network_root);self.bundle_nodes.append(node)
            # Static distributed highlights, intensity is actual filtered delivery.
            fractions=np.linspace(.12,.88,5)
            dashes=[(a+(b-a)*t,a+(b-a)*min(t+.07,1)) for t in fractions] if bundle.pre!=bundle.post else lines
            glow=geometry(f'transmission-{i}',dashes,kind='lines',color=(1,.55,.04,.7))
            glow.setRenderModeThickness(2.5);glow.reparentTo(self.network_root);self.glow_nodes.append(glow)
        for i,group in enumerate(self.grouped.groups):
            if group.centroid is None: self.group_nodes.append(None);continue
            node=geometry(f'group-{i}',[group.centroid],color=(.75,.50,.86,1))
            node.setRenderModeThickness(4);node.reparentTo(self.network_root);self.group_nodes.append(node)

    def _ui(self):
        for button in self.member_buttons: button.destroy()
        for node in self.ui_nodes: node.destroy() if hasattr(node,'destroy') else node.removeNode()
        self.ui_nodes=[];self.member_buttons=[]
        aspect=self.getAspectRatio(); self.ui_size=(self.win.getXSize(),self.win.getYSize())
        def text(value,x,y,size=.026,color=(.86,.83,.94,1)):
            node=OnscreenText(text=value,pos=(x*aspect,y),align=TextNode.ALeft,scale=size,fg=color,mayChange=True)
            self.ui_nodes.append(node);return node
        def button(value,x,y,fn):
            node=DirectButton(text=value,pos=(x*aspect,0,y),scale=min(.027,aspect*.018),command=fn,
                frameColor=(.10,.065,.16,1),text_fg=(.91,.86,.98,1),pad=(.6,.4),relief=0)
            self.ui_nodes.append(node);return node
        self.title=text('F L Y B R A I N   /   FLIGHT LAB',-.95,.91,.046)
        text('BANC v888  |  589 simulated cells  |  fixed source graph, modeled physiology',-.95,.85)
        self.activity_label=text('OBSERVATIONAL activity: teacher supplies flight commands',.04,.85,.026)
        text('01   FLOWER LANDING',-.95,.77,.03)
        self.view_title=text('02   NETWORK / CONNECTION DIAGRAM',.04,.77,.03)
        button('Follow / F',-.84,.68,self.toggle_follow);button('Reset camera',-.56,.68,self.reset_world)
        self.view_button=button('Network / N',.14,.68,self.toggle_view)
        self.vnc_button=button('Cord / V',.35,.68,self.toggle_vnc)
        for label,x in [('Front',.55),('Side',.67),('Top',.79)]: button(label,x,.68,lambda p=label.lower():self.preset(p))
        button('Reset / B',.92,.68,self.focus_brain)
        self.mode_button=button('Context / Tab',-.28,.68,self.toggle_mode)
        self.hud=text('',-.95,-.415,.025)
        self.coverage=text('',.04,-.415,.023)
        self.panels={}
        for name in ('help','details'):
            left,right,bottom,top=self.layout.rectangles[name]
            frame=DirectScrolledFrame(frameSize=(left*aspect,right*aspect,bottom,top),
                canvasSize=(0,(right-left)*aspect, -1,0),scrollBarWidth=.025,
                frameColor=(.025,.018,.043,1),borderWidth=(0,0))
            self.ui_nodes.append(frame);self.panels[name]=frame
        self.info=OnscreenText(parent=self.panels['details'].getCanvas(),text=self.info_text,pos=(.018,-.035),
            align=TextNode.ALeft,scale=.025,fg=(.86,.83,.94,1),mayChange=True,wordwrap=max(24,aspect*35))
        self.ui_nodes.append(self.info)
        helptext=('Drag viewport: orbit. Shift-drag: pan. Wheel: smooth zoom.\n'
            'Wheel anywhere in these panels: scroll text.\n1 / 2 / 3: front / side / top. N: Network / Anatomy.\n'
            'B: reset brain focus. V: nerve cord. F: follow fly.\nTab: anatomical context. C: selected source edges.\n'
            'R: new trial. Backspace: exact replay. Space: pause. Esc: exit.\n'
            'Manual: Up/Down power; W/S pitch; A/D roll; Q/E yaw; L legs.\n'
            'Network links join source-group centroids: not axon geometry.\n'
            'Warm light: modeled delivered signal, smoothed over 300 ms.\n'
            'Teacher/manual circuit activity is observational.\n'
            'Brightness does not establish behavioral importance.\n'
            '11 cells lack source positions; inspect their membership via groups.\n'
            'Readouts alone are trainable. Source graph and neuron rules are fixed.')
        self.help=OnscreenText(parent=self.panels['help'].getCanvas(),text=helptext,pos=(.018,-.035),
            align=TextNode.ALeft,scale=.025,fg=(.64,.60,.74,1),wordwrap=max(24,aspect*35))
        self.ui_nodes.append(self.help)
        self.panels['help']['canvasSize']=(0,.9*aspect,-max(.4,self.help.getTightBounds()[0].z* -1+.04),0)
        button('Pause / Space',-.79,-.92,self.toggle_pause);button('New flower / R',-.42,-.92,self.new_trial)
        button('Replay / Backspace',.03,-.92,self.replay);button('Exit / Esc',.78,-.92,self.userExit)
        button('All groups',.49,-.92,self.list_groups)
        self._set_info(self.info_text)
        if self.inspection:
            kind,value=self.inspection
            if kind=='group':self.select_group(value)
            elif kind=='member':self.select(value)
            else:self.list_groups()

    def _set_info(self,value):
        self.info_text=value
        if not hasattr(self,'info'):return
        self.info.setText(value)
        bounds=self.info.getTightBounds(); bottom=float(bounds[0].z) if bounds else -.1
        self.panels['details']['canvasSize']=(0,.9*self.getAspectRatio(),min(-.38,bottom-.12),0)
        self.panels['details'].verticalScroll['value']=0

    def _refresh_coverage(self):
        if hasattr(self,'coverage'):
            self.coverage.setText('96 source groups | purple: connections | warm: delivered signal' if self.network_view else
                f'Anatomy: {len(self.loaded)}/589 skeletons loaded; 4 unavailable')

    def _resize(self):
        self.aspect2d.setScale(1/self.getAspectRatio(),1,1)
        for camera,name in ((self.cam,'world'),(self.brain_camera,'brain')):
            left,right,bottom,top=self.layout.rectangles[name]
            camera.node().getDisplayRegion(0).setDimensions((left+1)/2,(right+1)/2,(bottom+1)/2,(top+1)/2)
            camera.node().getLens().setAspectRatio(self.win.getXSize()*(right-left)/(self.win.getYSize()*(top-bottom)))

    def _visibility(self):
        for p in self.project: p.hide() if self.network_view else p.show()
        self.network_root.show() if self.network_view else self.network_root.hide()
        for p in self.context:
            p.show() if self.whole else p.hide()
            for child in p.getChildren():
                if child.getName()=='source-reference-points': child.hide() if self.network_view else child.show()
        for i,b in enumerate(self.grouped.bundles):
            visible=self.vnc or all(self.grouped.groups[g].region!='ventral_nerve_cord' for g in (b.pre,b.post))
            if self.bundle_nodes[i]: self.bundle_nodes[i].show() if visible else self.bundle_nodes[i].hide()
        for group,node in zip(self.grouped.groups,self.group_nodes):
            if node: node.show() if self.vnc or group.region!='ventral_nerve_cord' else node.hide()

    def toggle_view(self):
        self.network_view=not self.network_view
        self.view_button['text']='Network / N' if self.network_view else 'Anatomy / N'
        self.view_title.setText('02   NETWORK / CONNECTION DIAGRAM' if self.network_view else '02   SOURCE ANATOMY')
        self._visibility()

    def toggle_mode(self):
        super().toggle_mode();self._visibility()

    def toggle_vnc(self):
        super().toggle_vnc()
        self.brain_motion.minimum=1.35 if self.vnc else .65
        self.brain_motion.reset(1.65 if self.vnc else .96)
        self.brain_target=np.array((.49,-.47,.15) if self.vnc else (.49,-.17,.145))
        self._visibility()

    def focus_brain(self):
        super().focus_brain();self.brain_motion.minimum=.65;self.brain_motion.reset(.96)

    def toggle_follow(self):
        super().toggle_follow();self.world_motion.reset(self.world_distance)

    def reset_world(self):
        self.follow=False;self.world_azimuth=-.6;self.world_elevation=.55
        self.world_motion.reset(1.05)

    def list_groups(self):
        self.inspection=('list',None)
        self._set_info('All source groups (including missing positions):\nSelect a group below to inspect members.')
        self._buttons([(f'{i:02d} {g.side} {g.region} ({len(g.members)})',lambda i=i:self.select_group(i)) for i,g in enumerate(self.grouped.groups)])

    def _buttons(self,rows):
        for button in self.member_buttons: button.destroy()
        self.member_buttons=[]
        bounds=self.info.getTightBounds();y=float(bounds[0].z)-.045
        for label,fn in rows:
            b=DirectButton(parent=self.panels['details'].getCanvas(),text=label,text_align=TextNode.ALeft,
                pos=(.018,0,y),scale=.023,command=fn,relief=0,frameColor=(.07,.04,.11,1),text_fg=(.84,.64,.95,1))
            self.member_buttons.append(b);y-=.037
        self.panels['details']['canvasSize']=(0,.9*self.getAspectRatio(),min(-.38,y-.05),0)

    def select_group(self,index):
        self.inspection=('group',index)
        self.selected_group=index;g=self.grouped.groups[index]
        incoming=[b for b in self.grouped.bundles if b.post==index];outgoing=[b for b in self.grouped.bundles if b.pre==index]
        self._set_info(f'Group {index:02d} | {g.side} {g.region}\n{len(g.members)} members | roles: {", ".join(g.roles)}\n'
            f'Incoming: {sum(len(b.edge_indices) for b in incoming)} edges from groups {", ".join(str(b.pre) for b in incoming)}\n'
            f'Outgoing: {sum(len(b.edge_indices) for b in outgoing)} edges to groups {", ".join(str(b.post) for b in outgoing)}\n'
            +('Source positions unavailable; no geometry fabricated.\n' if g.centroid is None else 'Centroid uses unchanged source coordinates.\n')+'Select a source member:')
        self._buttons([(id,lambda id=id:self.select(id)) for id in g.members])

    def select(self,id):
        self.inspection=('member',id)
        super().select(id);self._set_info(self.info.getText())
        self._buttons([('Back to group',lambda:self.select_group(self.selected_group))] if self.selected_group is not None else [])

    def pick(self,x,y):
        if self.layout.route(x,y)!='brain':return
        left,right,bottom,top=self.layout.rectangles['brain'];ndc=np.array(((x-left)/(right-left)*2-1,(y-bottom)/(top-bottom)*2-1))
        if self.network_view:
            candidates=[(i,g.centroid) for i,g in enumerate(self.grouped.groups) if g.centroid is not None and (self.vnc or g.region!='ventral_nerve_cord')]
        else: candidates=[(id,pos) for id,pos in self.root_positions.items() if (self.whole or id in self.nodes) and (self.vnc or self.records[id]['region']!='ventral_nerve_cord')]
        points=np.array([p for _,p in candidates]);matrix=np.array(self.brain_scene.getMat(self.brain_camera))
        local=np.c_[points,np.ones(len(points))]@matrix;fov=self.brain_camera.node().getLens().getFov()
        projected=local[:,[0,2]]/np.maximum(local[:,1:2],1e-9)/np.tan(np.radians(fov)/2)
        distances=np.linalg.norm(projected-ndc,axis=1);distances[local[:,1]<=0]=np.inf
        best=int(distances.argmin())
        if distances[best]<.08: (self.select_group if self.network_view else self.select)(candidates[best][0])

    def zoom_at(self,factor,x,y):
        panel=self.layout.route(x,y)
        if panel in ('help','details'):
            bar=self.panels[panel].verticalScroll
            bar['value']=float(np.clip(bar['value']+(.09 if factor>1 else -.09),0,1))
        elif panel=='brain':self.brain_motion.zoom(factor)
        elif panel=='world':self.world_motion.zoom(factor)

    def zoom(self,factor):
        if self.mouseWatcherNode and self.mouseWatcherNode.hasMouse():
            m=self.mouseWatcherNode.getMouse();self.zoom_at(factor,m.x,m.y)

    def start_drag(self):
        super().start_drag()
        if self.drag and self.layout.route(*self.drag[:2]) not in ('world','brain'): self.drag=None

    def start_pan(self):
        """Panda routes modified mouse events separately from plain mouse1."""
        self.keys.add('shift');self.start_drag()

    def _world_camera(self):
        super()._world_camera()
        if hasattr(self,'world_motion'):
            self.camera.setPos(self.camera.getPos()+Vec3(*self.world_motion.offset))

    def draw_snapshot(self,snapshot):
        super().draw_snapshot(snapshot)
        # Fixed 100 current units/s normalization per source edge, then mean
        # within each bundle. Brightness never rescales with the current frame.
        values=self.grouped.aggregate(snapshot.edge_activity)
        for b,node,value in zip(self.grouped.bundles,self.glow_nodes,values):
            if node is None:continue
            intensity=float(np.clip(value/(100*len(b.edge_indices)),0,1))
            visible=self.vnc or all(self.grouped.groups[g].region!='ventral_nerve_cord' for g in (b.pre,b.post))
            if intensity>.015 and visible:
                node.show();node.setColor(1,.40+.5*intensity,.035,.15+.75*intensity)
            else:node.hide()
        text=self.hud.getText().split('\n')
        self.hud.setText(f'{snapshot.mode.upper()} | seed {snapshot.seed} | {snapshot.time:.2f}s | dwell {snapshot.dwell:.2f}s'+(' | PAUSED' if self.paused else ''))
        self.activity_label.setText('OBSERVATIONAL activity: '+snapshot.mode+' supplies commands' if snapshot.mode in ('teacher','manual') else
            'Modeled signal transmission; brightness does not prove importance')
        self._refresh_coverage()

    def tick(self,task):
        if self.ui_size!=(self.win.getXSize(),self.win.getYSize()):self._ui();self._resize()
        dt=min(ClockObject.getGlobalClock().getDt(),.1)
        self.brain_motion.advance(dt);self.world_motion.advance(dt)
        self.distance=self.brain_motion.distance;self.world_distance=self.world_motion.distance
        if self.drag and self.mouseWatcherNode and self.mouseWatcherNode.hasMouse():
            m=self.mouseWatcherNode.getMouse();dx=m.x-self.drag[2];dy=m.y-self.drag[3]
            panel=self.layout.route(*self.drag[:2]);motion=self.brain_motion if panel=='brain' else self.world_motion
            if 'shift' in self.keys:
                camera=self.brain_camera if panel=='brain' else self.camera
                basis=camera.getQuat();motion.pan(np.array(basis.getRight())*(-dx*motion.distance)+np.array(basis.getUp())*(-dy*motion.distance))
            elif panel=='brain':self.azimuth-=dx*3;self.elevation=float(np.clip(self.elevation+dy*3,-1.56,1.569))
            else:self.world_azimuth-=dx*3;self.world_elevation=float(np.clip(self.world_elevation+dy*3,-1.4,1.4))
            self.drag=(*self.drag[:2],m.x,m.y)
        centre=np.array((.49,-.47,.15) if self.vnc else (.49,-.17,.145))
        self.brain_target=centre+self.brain_motion.offset;self._brain_camera();self._world_camera()
        drag=self.drag;self.drag=None
        result=super().tick(task);self.drag=drag
        return result

    def capture(self,directory):
        """Export opaque complete frames, including UI, from the actual renderer."""
        directory.mkdir(parents=True,exist_ok=True);self.paused=True
        self.load_geometry(100000)
        for _ in range(50):self.session.advance()
        def save(name):
            self.draw_snapshot(self.session.snapshot)
            for _ in range(3):self.taskMgr.step()
            picture=PNMImage();self.win.getScreenshot(picture);picture.removeAlpha()
            picture.write(str(directory/name))
        self.focus_brain()
        for name in ('front','side','top'):
            self.preset(name);save(f'brain-{name}.png')
        self.toggle_vnc();self.preset('front');save('brain-and-cord.png')
        self.focus_brain();self.toggle_view();save('anatomy.png');self.toggle_view()
        self.select_group(next(i for i,g in enumerate(self.grouped.groups) if len(g.members)>5))
        save('group-details.png')
