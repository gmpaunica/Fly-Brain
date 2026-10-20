"""Panda3D world and native anatomical viewer; drawing consumes snapshots only."""

from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
from pathlib import Path
import numpy as np
# Windows DLL initialization requires PyTorch before Panda's native core.
# Verified with two fresh interpreter import-order reproductions.
import torch
from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import DirectButton, DirectLabel, DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (loadPrcFileData, NodePath, Geom, GeomNode, GeomVertexData,
    GeomVertexFormat, GeomVertexWriter, GeomPoints, GeomLines, GeomTriangles,
    Camera, PerspectiveLens, TextNode, TransparencyAttrib, Vec3, Point3, ClockObject,
    AntialiasAttrib)

from .anatomy import CACHE, anatomical_positions, load_manifest
from .assets import fetch_skeleton
from .contracts import FlightCommand, CONTROL_DT
from .session import ActivityTrace, FlightSession
from .world import make_trial


def geometry(name: str,points,indices=None,kind='points',color=(.6,.3,.8,1),colors=None) -> NodePath:
    """Create a static render primitive without touching simulation state."""
    points=np.asarray(points).reshape(-1,3)
    format=GeomVertexFormat.getV3() if colors is None else GeomVertexFormat.getV3c4()
    data=GeomVertexData(name,format,Geom.UHStatic); data.setNumRows(len(points))
    # Panda's float32 vertex array accepts a contiguous buffer directly.
    if colors is None:
        data.modifyArray(0).modifyHandle().setData(np.asarray(points,dtype=np.float32).tobytes())
    else:
        vertex=GeomVertexWriter(data,'vertex'); paint=GeomVertexWriter(data,'color')
        for point,rgba in zip(points,colors): vertex.addData3f(*point); paint.addData4f(*rgba)
    primitive={'points':GeomPoints,'lines':GeomLines,'triangles':GeomTriangles}[kind](Geom.UHStatic)
    if indices is None: primitive.addConsecutiveVertices(0,len(points))
    else:
        values=np.asarray(indices,dtype=np.uint32).reshape(-1)
        primitive.setIndexType(Geom.NTUint32)
        primitive.modifyVertices().modifyHandle().setData(values.tobytes())
    primitive.closePrimitive(); geom=Geom(data); geom.addPrimitive(primitive)
    node=GeomNode(name); node.addGeom(geom); path=NodePath(node)
    if colors is None: path.setColor(*color)
    path.setLightOff(); path.setTwoSided(True)
    if kind=='points': path.setAntialias(AntialiasAttrib.MPoint)
    if color[3]<1: path.setTransparency(TransparencyAttrib.MAlpha)
    return path


def ellipsoid(parent,center,radii,color,name='surface'):
    rows,cols=12,24; vertices=[]; faces=[]
    for i in range(rows+1):
        theta=math.pi*i/rows
        for j in range(cols):
            phi=math.tau*j/cols
            vertices.append((math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)))
    for i in range(rows):
        for j in range(cols):
            a=i*cols+j; b=i*cols+(j+1)%cols
            faces.extend(((a,b,a+cols),(b,b+cols,a+cols)))
    unit=np.asarray(vertices); normals=unit/np.asarray(radii)
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    light=np.array((.4,-.6,.9)); light/=np.linalg.norm(light)
    shade=.32+.68*np.maximum(0,normals@light)
    colors=np.c_[shade[:,None]*np.asarray(color[:3]),np.full(len(unit),color[3])]
    vertices=unit*radii+center
    node=geometry(name,vertices,faces,'triangles',(1,1,1,color[3]),colors=colors)
    node.reparentTo(parent); return node


class FlightViewer(ShowBase):
    """Two independently navigable views sharing one fixed-step session."""

    def __init__(self,seed=10001,mode='teacher',checkpoint=None,offscreen=False):
        loadPrcFileData('', 'window-title FlyBrain | 3D flight and BANC anatomy\nwin-size 1440 900\nbackground-color 0.005 0.006 0.014 1\naudio-library-name null\nsync-video true\nshow-frame-rate-meter false')
        loadPrcFileData('',f'back-buffers {0 if offscreen else 1}')
        super().__init__(windowType='offscreen' if offscreen else None)
        # Anatomy is progressively loaded by this app. Each completed frame
        # must include resident UI/world geometry, even after cache paging.
        self.win.getGsg().setIncompleteRender(False)
        self.disableMouse(); self.offscreen=offscreen
        self.session=FlightSession(make_trial(seed),mode,checkpoint)
        self.manifest=self.session.manifest; self.nodes={n['id']:n for n in self.manifest['nodes']}
        self.trace=ActivityTrace(self.session.brain.ids)
        self.accumulator=0.; self.paused=False; self.keys=set(); self.drag=None
        self.whole=True; self.vnc=True; self.show_edges=False; self.selected=None; self.follow=False
        self.loaded=set(); self.failed=set(); self.coarse={}; self.details=None
        self.pool=ThreadPoolExecutor(max_workers=2); self.pending={}
        self.world_scene=self.render.attachNewNode('world')
        self.brain_scene=NodePath('anatomical-scene')
        self.brain_layers=[self.brain_scene.attachNewNode('brain'),self.brain_scene.attachNewNode('nerve-cord')]
        self.context=[p.attachNewNode('context') for p in self.brain_layers]
        self.project=[p.attachNewNode('project') for p in self.brain_layers]
        self.cam.node().getDisplayRegion(0).setDimensions(0,.49,.19,.88)
        self.camLens.setAspectRatio(1440*.49/(900*.69)); self.camLens.setNearFar(.001,10)
        self.camera.setPos(1.04,-.66,.67); self.camera.lookAt(.3,.3,.1)
        lens=PerspectiveLens(); lens.setFov(45); lens.setAspectRatio(1440*.51/(900*.69)); lens.setNearFar(.005,20)
        self.brain_camera=self.brain_scene.attachNewNode(Camera('anatomical-camera',lens))
        region=self.win.makeDisplayRegion(.49,1,.19,.88); region.setCamera(self.brain_camera)
        region.setSort(1)
        region.setClearColorActive(True); region.setClearColor((.001,.001,.004,1)); region.setClearDepthActive(True)
        self.brain_target=np.array((.49,-.47,.15)); self.azimuth=0.; self.elevation=math.pi/2-.01; self.distance=1.65
        self.world_azimuth=-.6; self.world_elevation=.55; self.world_distance=1.05
        self._world(); self._anatomy(); self._ui(); self.preset('front')
        for key in ('w','s','a','d','q','e','arrow_up','arrow_down','l'):
            self.accept(key,self.keys.add,[key]); self.accept(key+'-up',self.keys.discard,[key])
        for key,fn in (('r',self.new_trial),('backspace',self.replay),('space',self.toggle_pause),
                       ('escape',self.userExit),('v',self.toggle_vnc),('tab',self.toggle_mode),('c',self.toggle_edges)):
            self.accept(key,fn)
        for key,preset in (('1','front'),('2','side'),('3','top')): self.accept(key,self.preset,[preset])
        self.accept('mouse1',self.start_drag); self.accept('mouse1-up',self.stop_drag)
        self.accept('wheel_up',self.zoom,[.88]); self.accept('wheel_down',self.zoom,[1.14])
        self.accept('f',self.toggle_follow); self.accept('b',self.focus_brain)
        self.taskMgr.add(self.tick,'fixed-simulation-and-passive-views')

    def _ui(self):
        fontcolor=(.89,.88,.97,1)
        self.title=OnscreenText(text='F L Y B R A I N   /   FLIGHT LAB',pos=(-1.53,.91),align=TextNode.ALeft,
                                scale=.049,fg=fontcolor,mayChange=False)
        OnscreenText(text='BANC v888   |   589 simulated cells   |   3D research preview',pos=(-1.53,.84),
                     align=TextNode.ALeft,scale=.029,fg=(.55,.52,.7,1))
        OnscreenText(text='01    FLOWER LANDING',pos=(-1.53,.74),align=TextNode.ALeft,scale=.033,fg=(.84,.7,.9,1))
        OnscreenText(text='02    ANATOMICAL CIRCUIT',pos=(.035,.74),align=TextNode.ALeft,scale=.033,fg=(.84,.7,.9,1))
        def button(text,x,y,command):
            return DirectButton(text=text,pos=(x,0,y),scale=.030,command=command,
                                frameColor=(.085,.065,.14,1),text_fg=fontcolor,pad=(.5,.35),relief=0)
        self.mode_button=button('Whole brain',.25,.64,self.toggle_mode)
        self.vnc_button=button('Nerve cord: on',.66,.64,self.toggle_vnc)
        for text,x in (('Front',1.04),('Side',1.24),('Top',1.43)): button(text,x,.64,lambda p=text.lower():self.preset(p))
        button('Pause / Space',-1.28,-.88,self.toggle_pause)
        button('New flower / R',-.82,-.88,self.new_trial)
        button('Replay / Backspace',-.29,-.88,self.replay)
        button('Follow fly / F',-1.30,.64,self.toggle_follow)
        button('Brain focus / B',.27,.56,self.focus_brain)
        self.hud=OnscreenText(text='',pos=(-1.53,-.67),align=TextNode.ALeft,scale=.030,fg=fontcolor,mayChange=True)
        self.coverage=OnscreenText(text='',pos=(.035,-.67),align=TextNode.ALeft,scale=.025,fg=(.65,.57,.78,1),mayChange=True)
        self.info=OnscreenText(text='Click a neuron to inspect its source identity and role.',pos=(.035,-.78),
                               align=TextNode.ALeft,scale=.025,fg=fontcolor,mayChange=True,wordwrap=58)
        OnscreenText(text='Drag: orbit   Scroll: zoom   1 / 2 / 3: views   Tab: visibility   V: cord   C: source-edge diagram',
                     pos=(-1.53,-.97),align=TextNode.ALeft,scale=.023,fg=(.55,.52,.65,1))

    def _world(self):
        grid=[]
        for value in np.linspace(0,.6,13): grid.extend((((0,value,0),(.6,value,0)),((value,0,0),(value,.6,0))))
        geometry('arena-grid',grid,kind='lines',color=(.1,.17,.20,1)).reparentTo(self.world_scene)
        geometry('ground',[(0,0,-.0005),(.6,0,-.0005),(.6,.6,-.0005),(0,.6,-.0005)],
                 [(0,1,2),(0,2,3)],'triangles',(.016,.029,.033,1)).reparentTo(self.world_scene)
        self.flower=self.world_scene.attachNewNode('flower'); self._flower()
        self.fly=self.world_scene.attachNewNode('fly')
        ellipsoid(self.fly,(0,-.001,0),(.0017,.003,.0017),(.22,.15,.08,1),'abdomen')
        ellipsoid(self.fly,(0,.002,0),(.0019,.002,.0018),(.48,.34,.16,1),'thorax')
        ellipsoid(self.fly,(0,.004,0),(.0015,.0013,.0014),(.56,.16,.17,1),'head')
        self.wings=[]
        for sign in (-1,1):
            pivot=self.fly.attachNewNode('wing'); pivot.setX(sign*.001)
            ellipsoid(pivot,(sign*.004,-.0005,.0007),(.005,.0018,.00012),(.53,.73,.82,.52),'wing-surface')
            self.wings.append(pivot)
        self.legs=self.fly.attachNewNode('legs')
        leglines=[]
        for side in (-1,1):
            for y in (-.002,0,.002): leglines.append(((side*.001,y,0),(side*.003,y-.001,-.003)))
        geometry('leg-lines',leglines,kind='lines',color=(.8,.62,.34,1)).reparentTo(self.legs)
        self.fly_marker=geometry('fly-location',[(0,0,0)],color=(1,.66,.18,1)); self.fly_marker.setRenderModeThickness(5)
        self.fly_marker.reparentTo(self.world_scene)
        self.flight_path=[]; self.path_node=None

    def _flower(self):
        self.flower.getChildren().detach()
        x,y,z=self.session.world.case.flower
        ellipsoid(self.flower,(x,y,z/2),(.002,.002,z/2),(.12,.35,.19,1),'stem')
        for i in range(8):
            a=i*math.tau/8
            ellipsoid(self.flower,(x+.012*math.cos(a),y+.012*math.sin(a),z-.004),(.013,.013,.004),(.61,.13,.37,1),'petal')
        points=[(x,y,z)]+[(x+.015*math.cos(i*math.tau/48),y+.015*math.sin(i*math.tau/48),z) for i in range(48)]
        geometry('nectar-surface',points,[(0,i+1,(i+1)%48+1) for i in range(48)],'triangles',(1,.64,.09,1)).reparentTo(self.flower)

    def _anatomy(self):
        with gzip.open(CACHE/'atlas.json.gz','rt',encoding='utf-8') as f: atlas=json.load(f)
        self.atlas=atlas; self.records={r['id']:r for r in atlas['records']}
        coverage_path=CACHE/'coverage.json'
        self.source_missing=set()
        if coverage_path.exists():
            self.source_missing={r['id'] for r in json.loads(coverage_path.read_text())['skeletons'] if r['status']!='available'}
        self.root_positions={r['id']:anatomical_positions(r['position_nm']) for r in atlas['records'] if r['position_nm'] is not None}
        for cord in (False,True):
            rows=[r for r in atlas['records'] if r['position_nm'] is not None and (r['region']=='ventral_nerve_cord')==cord]
            points=anatomical_positions([r['position_nm'] for r in rows])
            node=geometry('source-reference-points',points,color=(.38,.16,.50,.34))
            node.setRenderModeThickness(1); node.reparentTo(self.context[int(cord)])
            selected=[n for n in self.manifest['nodes'] if n['position_nm'] is not None and (n['metadata']['region']=='ventral_nerve_cord')==cord]
            node=geometry('project-reference-points',anatomical_positions([n['position_nm'] for n in selected]),color=(.83,.43,.87,1))
            node.setRenderModeThickness(3); node.reparentTo(self.project[int(cord)])
            path=CACHE/'regions'/('nerve_cord.npz' if cord else 'brain.npz')
            if path.exists():
                with np.load(path) as mesh:
                    node=geometry('registered-region',mesh['xyz'],mesh['faces'],'triangles',(.29,.1,.45,.12))
                    node.setRenderModeWireframe(); node.reparentTo(self.context[int(cord)])
        self.activity_nodes=[]
        for node in self.manifest['nodes']:
            if node['position_nm'] is None: self.activity_nodes.append(None); continue
            point=geometry(node['id'],[anatomical_positions(node['position_nm'])],color=(1,.48,.05,1))
            point.setRenderModeThickness(5); point.reparentTo(self.project[int(node['metadata']['region']=='ventral_nerve_cord')])
            self.activity_nodes.append(point)
        self.edge_node=self.brain_scene.attachNewNode('selected-source-edges')
        self.selection_nodes=[p.attachNewNode('selection') for p in self.brain_layers]
        self._refresh_coverage()

    def _refresh_coverage(self):
        if hasattr(self,'coverage'):
            missing=self.atlas['missing_positions']; untyped=self.atlas['untyped_records']
            self.coverage.setText(f'{len(self.loaded)}/589 project skeletons loaded; {len(self.source_missing)} source exports unavailable\n'
                f'{missing:,} positions missing; {untyped:,} untyped atlas records\n'
                f'{len(self.atlas["records"])-missing:,} reference points; unused-cell detail loads on selection.')

    def load_geometry(self,limit=4):
        """Progressive cached LOD; absent detailed cells fetched on selection."""
        for node in self.manifest['nodes']:
            id=node['id']; path=CACHE/'skeletons'/f'{id}.npz'
            if id in self.loaded or not path.exists() or not path.with_suffix('.json').exists(): continue
            with np.load(path) as data: lines=data['coarse']
            parts=[]
            for cord in (False,True):
                keep=(lines.mean(1)[:,1]<-.43)==cord
                geom=geometry('skeleton-'+id,lines[keep],kind='lines',color=(.59,.27,.75,.50))
                geom.setRenderModeThickness(1); geom.reparentTo(self.project[int(cord)]); parts.append(geom)
            self.coarse[id]=parts; self.loaded.add(id); limit-=1
            if limit<=0: break
        self._refresh_coverage()
        if self.selected in self.pending and self.pending[self.selected].done():
            record=self.pending.pop(self.selected).result()
            if record['status']=='available': self.select(self.selected)
            else: self.info.setText(self.info.getText()+'\nSource skeleton unavailable; reference point retained.')

    def preset(self,name):
        # BANC X-Y is the preparation plane; anterior is native -Y. These
        # documented camera presets never transform an individual neuron.
        self.azimuth,self.elevation={'front':(0,math.pi/2-.001),'side':(math.pi/2,.05),'top':(0,.05)}[name]
        self._brain_camera()

    def _brain_camera(self):
        offset=self.distance*np.array((math.sin(self.azimuth)*math.cos(self.elevation),
            -math.cos(self.azimuth)*math.cos(self.elevation),math.sin(self.elevation)))
        self.brain_camera.setPos(*(self.brain_target+offset)); self.brain_camera.lookAt(Point3(*self.brain_target),Vec3(0,1,0) if self.elevation>1.56 else Vec3(0,0,1))

    def toggle_mode(self):
        self.whole=not self.whole
        for node in self.context: node.show() if self.whole else node.hide()
        for node in self.selection_nodes:
            node.show() if self.whole or self.selected in self.nodes else node.hide()
        self.mode_button['text']='Whole brain' if self.whole else 'Project neurons only'

    def toggle_vnc(self):
        self.vnc=not self.vnc; self.brain_layers[1].show() if self.vnc else self.brain_layers[1].hide()
        self.vnc_button['text']='Nerve cord: on' if self.vnc else 'Nerve cord: off'
        if self.vnc:
            self.brain_target=np.array((.49,-.47,.15)); self.distance=1.65; self._brain_camera()
        if self.selected and self.show_edges: self.select(self.selected)

    def focus_brain(self):
        self.brain_target=np.array((.49,-.17,.145)); self.distance=.96
        if self.vnc: self.toggle_vnc()
        self._brain_camera()

    def toggle_follow(self):
        self.follow=not self.follow; self.world_distance=.14 if self.follow else 1.05
        self._world_camera()

    def toggle_edges(self):
        self.show_edges=not self.show_edges
        if self.selected: self.select(self.selected)

    def select(self,id):
        if id not in self.records: return
        self.selected=id; row=self.records[id]; project=self.nodes.get(id)
        for layer in self.selection_nodes:
            layer.show() if self.whole or project else layer.hide()
        role=', '.join(project['roles']) if project else 'context only; not simulated'
        evidence=project['evidence'] if project else 'BANC source annotation'
        self.info.setText(f'{id}  |  {row["type"] or "untyped"}\n{role}\nEvidence: {evidence}\n'
                          'Positions are source reference points, not inferred soma centres.')
        for layer in self.selection_nodes: layer.getChildren().detach()
        self.edge_node.getChildren().detach()
        path=CACHE/'skeletons'/f'{id}.npz'
        if path.exists():
            with np.load(path) as data:
                lines=data['xyz'][data['edges']]
                for cord in (False,True):
                    keep=(lines.mean(1)[:,1]<-.43)==cord
                    node=geometry('selected-full-morphology',lines[keep],kind='lines',color=(.98,.60,.93,.85))
                    node.reparentTo(self.selection_nodes[int(cord)])
        elif id in self.source_missing:
            self.info.setText(self.info.getText()+'\nMorphology missing from public v888 SWC exports; source point retained.')
        elif id not in self.pending:
            self.pending[id]=self.pool.submit(fetch_skeleton,id)
        if self.show_edges and project:
            lines=[(self.root_positions[e['pre']],self.root_positions[e['post']]) for e in self.manifest['edges']
                   if id in (e['pre'],e['post']) and e['pre'] in self.root_positions and e['post'] in self.root_positions
                   and (self.vnc or all(self.root_positions[v][1]>=-.43 for v in (e['pre'],e['post'])))]
            if lines: geometry('directed-source-edge-diagram',lines,kind='lines',color=(.9,.45,.75,.35)).reparentTo(self.edge_node)
            incoming=sum(e['post']==id for e in self.manifest['edges']); outgoing=sum(e['pre']==id for e in self.manifest['edges'])
            self.info.setText(self.info.getText()+f'\nSource-edge diagram: {incoming} incoming / {outgoing} outgoing (not axon geometry).')

    def pick(self,x,y):
        if x<-.02 or not -.62<y<.76: return
        ndc=np.array(((x+.02)/1.02*2-1,(y+.62)/1.38*2-1))
        ids=[id for id in self.root_positions if (self.whole or id in self.nodes)
             and (self.vnc or self.records[id]['region']!='ventral_nerve_cord')]
        points=np.array([self.root_positions[id] for id in ids])
        transform=np.array(self.brain_scene.getMat(self.brain_camera))
        local=np.c_[points,np.ones(len(points))]@transform
        fov=self.brain_camera.node().getLens().getFov()
        projected=local[:,[0,2]]/np.maximum(local[:,1:2],1e-9)/np.tan(np.radians(fov)/2)
        distances=np.linalg.norm(projected-ndc,axis=1); distances[local[:,1]<=0]=np.inf
        i=int(np.argmin(distances))
        if distances[i]<.05: self.select(ids[i])

    def start_drag(self):
        if self.mouseWatcherNode and self.mouseWatcherNode.hasMouse():
            m=self.mouseWatcherNode.getMouse(); self.drag=(m.x,m.y,m.x,m.y)

    def stop_drag(self):
        if self.drag and self.mouseWatcherNode and self.mouseWatcherNode.hasMouse():
            m=self.mouseWatcherNode.getMouse()
            if math.hypot(m.x-self.drag[0],m.y-self.drag[1])<.008: self.pick(m.x,m.y)
        self.drag=None

    def zoom(self,factor):
        if self.mouseWatcherNode and self.mouseWatcherNode.hasMouse() and self.mouseWatcherNode.getMouse().x<-.02:
            self.world_distance=float(np.clip(self.world_distance*factor,.08,2)); self._world_camera()
        else: self.distance=float(np.clip(self.distance*factor,.1,5)); self._brain_camera()

    def _world_camera(self):
        a,e=self.world_azimuth,self.world_elevation
        target=np.array(self.session.snapshot.position) if self.follow else np.array((.3,.3,.1))
        self.camera.setPos(*(target+self.world_distance*np.array((math.sin(a)*math.cos(e),-math.cos(a)*math.cos(e),math.sin(e)))))
        self.camera.lookAt(Point3(*target))

    def toggle_pause(self): self.paused=not self.paused

    def replay(self):
        self.session.replay(); self.trace=ActivityTrace(self.session.brain.ids); self.accumulator=0.
        self.flight_path=[]

    def new_trial(self):
        self.session.new_trial(); self._flower(); self.trace=ActivityTrace(self.session.brain.ids)
        self.accumulator=0.; self.flight_path=[]

    def draw_snapshot(self,snapshot):
        """Passive rendering: immutable body/activity data and simulation time."""
        self.fly.setPos(*snapshot.position)
        r,p,y=snapshot.orientation; self.fly.setHpr(-math.degrees(y),-math.degrees(p),math.degrees(r))
        self.fly_marker.setPos(*snapshot.position)
        if self.follow: self._world_camera()
        for sign,wing in zip((-1,1),self.wings):
            # 12 Hz is a visible slowed proxy, below the 50 Hz snapshot Nyquist
            # rate; 25 Hz would alias to a motionless wing at every snapshot.
            wing.setR(sign*35*math.sin(snapshot.time*math.tau*12)*snapshot.command.power)
        self.legs.setSz(.25+.75*snapshot.command.legs)
        self.trace.update(snapshot.time,snapshot.spike_counts)
        for node,value in zip(self.activity_nodes,self.trace.values(snapshot.time)):
            if node is not None:
                if value>.03: node.show(); node.setColor(1,.32+.5*float(value),.07,float(value)); node.setRenderModeThickness(3+3*float(value))
                else: node.hide()
        mode={'teacher':'HAND-WRITTEN TEACHER | neural activity is observational',
              'manual':'MANUAL | Up/Down power, WASD tilt, Q/E turn, L legs',
              'fixed':'FIXED NEURAL DECODER','trained':'LEARNED READOUT (check recorded gates)',
              'untrained':'UNTRAINED NEURAL READOUT'}[snapshot.mode]
        self.hud.setText(f'{mode}\nTrial {snapshot.seed}  |  {snapshot.time:5.2f} s  |  height {snapshot.position[2]*100:.1f} cm\n'
                         f'Power {snapshot.command.power:.2f}  |  contact {"food" if snapshot.food_contact else "none"}  |  dwell {snapshot.dwell:.2f} s'
                         +('  |  IMPACT' if snapshot.crashed else '')+('  |  PAUSED' if self.paused else ''))

    def tick(self,task):
        if not self.paused:
            self.accumulator+=min(ClockObject.getGlobalClock().getDt(),.1)
            while self.accumulator>=CONTROL_DT:
                power=.5+.2*(('arrow_up' in self.keys)-('arrow_down' in self.keys))
                command=FlightCommand(power,.4*(('w' in self.keys)-('s' in self.keys)),
                    .4*(('d' in self.keys)-('a' in self.keys)),.3*(('e' in self.keys)-('q' in self.keys)),float('l' in self.keys))
                self.session.advance(command); self.trace.update(self.session.snapshot.time,self.session.snapshot.spike_counts)
                self.accumulator-=CONTROL_DT
        if self.drag and self.mouseWatcherNode and self.mouseWatcherNode.hasMouse():
            m=self.mouseWatcherNode.getMouse(); dx=m.x-self.drag[2]; dy=m.y-self.drag[3]
            if self.drag[0]>=-.02:
                self.azimuth-=dx*3; self.elevation=float(np.clip(self.elevation+dy*3,-1.56,1.569)); self._brain_camera()
            else:
                self.world_azimuth-=dx*3; self.world_elevation=float(np.clip(self.world_elevation+dy*3,-1.4,1.4)); self._world_camera()
            self.drag=(*self.drag[:2],m.x,m.y)
        self.load_geometry(); self.draw_snapshot(self.session.snapshot)
        return task.cont

    def capture(self,directory: Path):
        """Export reproducible saved views, distinct from a live input review."""
        directory.mkdir(parents=True,exist_ok=True); self.paused=True
        self.load_geometry(100000)
        for _ in range(50): self.session.advance()
        self.draw_snapshot(self.session.snapshot)
        self.preset('front')
        for _ in range(3): self.taskMgr.step()
        self.win.saveScreenshot(str(directory/'brain-and-cord.png'))
        self.focus_brain(); self.toggle_follow(); self.draw_snapshot(self.session.snapshot)
        for name in ('front','side','top'):
            self.preset(name)
            for _ in range(3): self.taskMgr.step()
            self.win.saveScreenshot(str(directory/f'brain-{name}.png'))
        self.toggle_mode(); self.preset('front')
        for _ in range(3): self.taskMgr.step()
        self.win.saveScreenshot(str(directory/'project-only.png'))

    def destroy(self):
        self.pool.shutdown(wait=False,cancel_futures=True)
        super().destroy()
