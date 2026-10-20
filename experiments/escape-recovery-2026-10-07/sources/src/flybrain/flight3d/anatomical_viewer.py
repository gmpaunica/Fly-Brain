"""Batched source anatomy colored by actual whole-neuron firing on black."""
from collections import deque
import gzip
import json
import math
from pathlib import Path
import time
import numpy as np
import torch  # Windows native DLL initialization precedes Panda3D.
from panda3d.core import ClockObject, Point3, Vec3, PNMImage

from .anatomy import CACHE, anatomical_positions
from .anatomical_display import MorphologyCache
from .activity import DisplayFrame, SpikeRecorder, pulse_strengths
from .branch_activity import BranchActivityTable
from .activity_classes import DisplayActivityIndex
from .contracts import CONTROL_DT, FlightCommand
from .network_viewer import FlightViewer as NavigableViewer
from .viewer import geometry
from .simulation_pump import SimulationPump
from .brain_controls import BrainControls
from .brain_camera import BrainCamera

LOAD_BUDGET_SECONDS=.002
BATCH_SEGMENTS=50000


class FlightViewer(BrainControls, NavigableViewer):
    """Reuse navigation/UI without constructing the rejected centroid network.

    Source geometry and neuron-level illumination have separate scene roots.
    All physics/neural stepping remains in the existing fixed-step session.
    """
    def __init__(self,*args,**kwargs):
        self.settings_open=False;self.views_open=False
        self.enabled_classes=frozenset({'odor'});self.activity_scope='whole';self.sensory_expanded=True
        self._context_bounds=[[],[]];self._branch_bounds=[[],[]]
        self.morphologies=MorphologyCache()
        self.source_batches=[];self.surface_nodes=[];self.detail_view=False
        self._geometry_pending={};self._batch_parts=[[],[]];self._batch_owners=[[],[]];self._batch_counts=[0,0]
        self._resident_owners=[set(),set()];self._known_branch_owners=[set(),set()];self._display_frame=None
        self._deferred_cord=deque();self._selection_parts=deque();self._selection_pending=None
        self._selection_cord=deque()
        self.visible_activity_ids=()
        self._activity_token=None;self.activity_updates=0;self._body_token=None;self.neuron_page=0
        self._pick_parts=[];self._pick_points=None;self._pick_ids=None
        self._coverage_text=None;self._load_finished=False
        self._selection_info_token=None
        super().__init__(*args,**kwargs)
        self.ignore('n');self.ignore('c')
        self.vnc=False;self.brain_layers[1].hide()
        self.network_view=False
        self.info_text='Click a source neuron to inspect its branches, identity and evidence.'
        self._ui();self._resize();self._visibility()
        self.brain_motion=BrainCamera();self._brain_camera()
        self.simulation=SimulationPump(self.session,SpikeRecorder(self.session.snapshot))
        self.draw_frame(self.simulation.display_frame)

    def _anatomy(self):
        self.activity_index=DisplayActivityIndex(self.manifest)
        if self.activity_index.ids!=self.session.brain.ids:raise ValueError('display/brain neuron order mismatch')
        self.activity_mask=self.activity_index.mask(self.enabled_classes)
        self.scope_count=len(self.activity_index.ids);self.enabled_firing=0
        self.activity_table=BranchActivityTable(self.session.brain.ids,np.asarray(self.activity_index.activation_colors))
        self._owner_index={id:i for i,id in enumerate(self.session.brain.ids)}
        # The retained graph is fixed. Inspector refreshes need these counts,
        # not a fresh scan of every edge on every display frame.
        self._edge_counts={id:[0,0,0] for id in self.session.brain.ids}
        for edge in self.manifest['edges']:
            self._edge_counts[edge['post']][0]+=1
            self._edge_counts[edge['pre']][1]+=1
            if edge['pre']!=edge['post']:self._edge_counts[edge['post']][2]+=1
        with gzip.open(CACHE/'atlas.json.gz','rt',encoding='utf-8') as stream:self.atlas=json.load(stream)
        self.records={r['id']:r for r in self.atlas['records']}
        self.root_positions={r['id']:anatomical_positions(r['position_nm']) for r in self.atlas['records'] if r['position_nm'] is not None}
        coverage=json.loads((CACHE/'coverage.json').read_text())
        self.source_missing={r['id'] for r in coverage['skeletons'] if r['status']!='available'}
        # Faint registered context preserves the optic lobes and central outline.
        positions=anatomical_positions([r['position_nm'] for r in self.atlas['records'] if r['position_nm'] is not None])
        for cord in (False,True):
            points=positions[(positions[:,1]<-.43)==cord]
            points=points[::max(1,len(points)//60000)]
            self._context_bounds[int(cord)].extend((points.min(axis=0),points.max(axis=0)))
            context=geometry('registered-source-context',points,color=(.24,.35,.58,.32))
            context.setRenderModeThickness(1);context.reparentTo(self.context[int(cord)])
            with np.load(CACHE/'regions'/('nerve_cord.npz' if cord else 'brain.npz')) as mesh:
                self._context_bounds[int(cord)].extend((mesh['xyz'].min(axis=0),mesh['xyz'].max(axis=0)))
                surface=geometry('registered-soft-surface',mesh['xyz'],mesh['faces'],'triangles',(.18,.23,.40,.12))
            surface.setTwoSided(False);surface.setDepthWrite(False)
            surface.reparentTo(self.context[int(cord)]);self.surface_nodes.append(surface)
        self.activity_nodes=[]
        self.edge_node=self.brain_scene.attachNewNode('selected-source-edges')
        self.selection_nodes=[p.attachNewNode('selected-source-morphology') for p in self.brain_layers]
        self._geometry_queue=deque(n['id'] for n in self.manifest['nodes'] if
            (CACHE/'skeletons'/f"{n['id']}.npz").exists() and
            (CACHE/'skeletons'/f"{n['id']}.json").exists())
        self._schedule_geometry()

    def _schedule_geometry(self):
        while self._geometry_queue and len(self._geometry_pending)<2:
            id=self._geometry_queue.popleft()
            self._geometry_pending[id]=self.pool.submit(self.morphologies.coarse,id)

    def _make_source_batch(self,lines,owners,cord):
        if not len(lines):return
        if cord and not self.vnc:
            self._deferred_cord.append((lines,owners));return
        node=self.activity_table.make_batch(lines,owners,'batched-source-branches')
        node.setRenderModeThickness(1);node.reparentTo(self.project[int(cord)])
        self.source_batches.append(node)
        self._resident_owners[int(cord)].update(int(i) for i in np.unique(owners))
        self._activity_token=None

    def _append_source(self,lines,cord,owner):
        index=int(cord)
        if len(lines):
            self._known_branch_owners[index].add(owner)
            self._branch_bounds[index].extend((lines.reshape(-1,3).min(axis=0),lines.reshape(-1,3).max(axis=0)))
        while len(lines):
            take=min(len(lines),BATCH_SEGMENTS-self._batch_counts[index])
            self._batch_parts[index].append(lines[:take])
            self._batch_owners[index].append(np.full(take,owner,dtype=np.int32))
            self._batch_counts[index]+=take;lines=lines[take:]
            if self._batch_counts[index]==BATCH_SEGMENTS:
                self._make_source_batch(np.concatenate(self._batch_parts[index]),np.concatenate(self._batch_owners[index]),cord)
                self._batch_parts[index]=[];self._batch_owners[index]=[];self._batch_counts[index]=0

    def load_geometry(self,limit=4):
        """Prepare cached files on two I/O threads; bound main-thread ingestion.

        A single render-buffer upload is indivisible. Each upload is capped at
        50,000 segments. Explicit capture/benchmark drains the queue fully.
        """
        drain=limit>=100000;deadline=time.perf_counter()+LOAD_BUDGET_SECONDS;done=0
        while self._geometry_pending and (drain or (done<limit and time.perf_counter()<deadline)):
            id,future=next(iter(self._geometry_pending.items()))
            if not future.done() and not drain:break
            lines=future.result();del self._geometry_pending[id]
            cord=lines.mean(axis=1)[:,1]<-.43
            self._append_source(lines[~cord],False,self._owner_index[id])
            self._append_source(lines[cord],True,self._owner_index[id])
            # Pick real branch samples, including cells without a root position.
            samples=lines[::max(1,len(lines)//48)].mean(axis=1) if len(lines) else np.empty((0,3))
            self._pick_parts.append((id,samples));self._pick_points=None
            self.loaded.add(id);done+=1;self._schedule_geometry()
        if not self._geometry_queue and not self._geometry_pending and not self._load_finished:
            for cord in (False,True):
                index=int(cord)
                if self._batch_parts[index]:self._make_source_batch(np.concatenate(self._batch_parts[index]),np.concatenate(self._batch_owners[index]),cord)
            self._batch_parts=[[],[]];self._batch_owners=[[],[]];self._batch_counts=[0,0];self._load_finished=True
        while self.vnc and self._deferred_cord and (drain or time.perf_counter()<deadline):
            lines,owners=self._deferred_cord.popleft()
            self._make_source_batch(lines,owners,True)
        while self.vnc and self._selection_cord:
            self._selection_parts.append((True,self._selection_cord.popleft()))
        if self._selection_pending:
            id,future=self._selection_pending
            if future.done():
                lines=future.result();self._selection_pending=None
                if id==self.selected:
                    self._prepare_selection(lines)
                    self._refresh_selected_info(self._display_frame)
        while self._selection_parts and (drain or time.perf_counter()<deadline):
            cord,lines=self._selection_parts.popleft()
            node=geometry('selected-full-source-branches',lines,kind='lines',color=(.36,.86,1,.88))
            node.setRenderModeThickness(1.8);node.reparentTo(self.selection_nodes[int(cord)])
        if done:self._activity_token=None
        self._refresh_coverage()

    def _refresh_coverage(self):
        if not hasattr(self,'coverage'):return
        frame=self._display_frame
        latest=sum(count>0 for count in frame.snapshot.spike_counts) if frame else 0
        total=sum(count>0 for count in frame.cumulative_counts) if frame else 0
        size=len(self.activity_index.ids)
        value=(f'{len(self.loaded)}/585 skeletons | 4 unavailable | Full model: {latest}/{size} firing, {total} fired since reset\n'
               f'Scope {self.scope_count} | enabled {sum(self.activity_mask)}: {self.enabled_firing} firing | {len(self.visible_activity_ids)} visible colored pulses')
        if value!=self._coverage_text:self.coverage.setText(value);self._coverage_text=value

    def _visibility(self):
        for context in self.context:context.show() if self.whole else context.hide()
        for part in self.project:part.show()
        self.brain_layers[1].show() if self.vnc else self.brain_layers[1].hide()
        for part in self.selection_nodes:part.show()
        self._activity_token=None

    def toggle_view(self):
        self.detail_view=not self.detail_view
        self._visibility()
        if self.settings_open:self._restore_inspector()

    def toggle_mode(self):
        self.whole=not self.whole;self._visibility()
        if self.settings_open:self._restore_inspector()

    def _prepare_selection(self,lines):
        for root in self.selection_nodes:root.getChildren().detach()
        self._selection_parts.clear();self._selection_cord.clear()
        for cord in (False,True):
            part=lines[(lines.mean(axis=1)[:,1]<-.43)==cord]
            for start in range(0,len(part),BATCH_SEGMENTS):
                chunk=part[start:start+BATCH_SEGMENTS]
                if cord and not self.vnc:self._selection_cord.append(chunk)
                else:self._selection_parts.append((cord,chunk))

    def select(self,id):
        if id not in self.records:return
        self.settings_open=False
        self.selected=id;self.inspection=('member',id)
        self._refresh_activity_choices()
        path=CACHE/'skeletons'/f'{id}.npz'
        self._selection_pending=(id,self.pool.submit(self.morphologies.full,id)) if path.exists() else None
        for root in self.selection_nodes:root.getChildren().detach()
        self._selection_parts.clear();self._selection_cord.clear()
        self._show_selection_info();self._draw_connections()

    def _selection_info(self,frame):
        id=self.selected;record=self.records[id];project=self.nodes.get(id)
        roles=', '.join(project['roles']) if project else 'background only; not simulated'
        evidence=project['evidence'] if project else 'source annotation'
        incoming,outgoing,nonself=self._edge_counts.get(id,(0,0,0))
        text=f'{id} | {record["type"] or "untyped"}\nModel role: {roles}\nSource: BANC v888 | {record["region"]}\nEvidence: {evidence}\nRetained edges: {incoming} incoming / {outgoing} outgoing\n'
        if project and not nonself:
            text+='No non-self incoming connections in this subset; self-connections are excluded from dynamics. Sensory inputs may receive external stimulation.\n'
        if project and frame:
            index=self._owner_index[id];last=frame.last_active_times[index]
            age='not since reset' if last is None else f'{max(0.,frame.snapshot.time-last):.3f} s'
            text+=(f'Latest interval spikes: {frame.snapshot.spike_counts[index]}\n'
                   f'Spikes since reset: {frame.cumulative_counts[index]}\n'
                   f'Recent firing rate: {frame.firing_rates_hz[index]:.2f} spikes/s (200 ms average)\n'
                   f'Time since last activity: {age}\n')
        elif not project:text+='Spike activity: not simulated.\n'
        if (CACHE/'skeletons'/f'{id}.npz').exists():
            text+=('Full source morphology loading from local cache.' if self._selection_pending else 'Full source morphology loaded from local cache.')+' Cyan is selection.'
            if project:
                owner=self._owner_index[id]
                if id not in self.loaded:text+=' Project branches loading.'
                elif owner not in self._resident_owners[0] and (not self.vnc or owner not in self._resident_owners[1]):
                    if not self.vnc and owner in self._known_branch_owners[1] and owner not in self._known_branch_owners[0]:
                        text+=' Only nerve-cord branches are available; Show nerve cord to display them. Visibility and framing remain your choice.'
                    else:text+=' No branches currently visible: empty or awaiting batch upload.'
        else:text+='Full morphology unavailable in local source cache; source identity retained.'
        text+='\nShow connections: straight links represent retained connections, not anatomical branches.'
        return text

    def _show_selection_info(self):
        self._set_info(self._selection_info(self._display_frame))
        self._buttons([('Focus selected',self.focus_selected),('Clear selection',self.clear_selection),
            ('Show connections: '+('on' if self.show_edges else 'off'),self.toggle_edges),
            ('Show nerve cord: '+('on' if self.vnc else 'off'),self.toggle_vnc),
            ('Browse neurons',self.list_groups)])
        self._selection_info_token=self._inspector_token(self._display_frame)

    def _inspector_token(self,frame):
        return (self.selected,frame.episode_id if frame else None,frame.step_index if frame else None,
                self.vnc,bool(self._selection_pending),len(self.loaded),len(self.source_batches))

    def _refresh_selected_info(self,frame):
        """Refresh values without rebuilding controls or losing panel scroll."""
        if not self.selected or self.settings_open or self.inspection!=('member',self.selected):return
        token=self._inspector_token(frame)
        if token==self._selection_info_token:return
        self._selection_info_token=token
        text=self._selection_info(frame)
        if text==self.info_text:return
        panel=self.panels['details'];scroll=panel.verticalScroll['value']
        self._set_info(text)
        bounds=self.info.getTightBounds();y=float(bounds[0].z)-.045
        for button in self.member_buttons:button.setZ(y);y-=.037
        panel['canvasSize']=(0,.9*self.getAspectRatio(),min(-.38,y-.05),0)
        panel.verticalScroll['value']=scroll

    def _draw_connections(self):
        self.edge_node.getChildren().detach()
        if not self.selected or not self.show_edges:return
        lines=[(self.root_positions[e['pre']],self.root_positions[e['post']]) for e in self.manifest['edges']
            if self.selected in (e['pre'],e['post']) and all(v in self.root_positions for v in (e['pre'],e['post']))
            and (self.vnc or all(self.root_positions[v][1]>=-.43 for v in (e['pre'],e['post'])))]
        if lines:geometry('selected-source-connection-diagram',lines,kind='lines',color=(.6,.35,.8,.4)).reparentTo(self.edge_node)

    def toggle_edges(self):
        if not self.selected:return
        self.show_edges=not self.show_edges;self._draw_connections();self._show_selection_info()

    def clear_selection(self):
        self.selected=None;self.inspection=None;self.show_edges=False;self._selection_pending=None
        self._refresh_activity_choices()
        self._selection_parts.clear();self._selection_cord.clear()
        for root in self.selection_nodes:root.getChildren().detach()
        self.edge_node.getChildren().detach();self._restore_inspector()

    def focus_selected(self):
        if not self.selected:return
        path=CACHE/'skeletons'/f'{self.selected}.npz'
        if path.exists():
            points=self.morphologies.coarse(self.selected).reshape(-1,3)
            if not self.vnc:points=points[points[:,1]>=-.43]
        else:
            point=self.root_positions.get(self.selected)
            points=np.array([point]) if point is not None else np.empty((0,3))
        if len(points):
            self.brain_motion.fit(np.array((points.min(axis=0),points.max(axis=0))),
                tuple(self.brain_camera.node().getLens().getFov()))
        else:self._set_info(self.info_text+'\nNo visible source geometry to focus; camera unchanged.')

    def list_groups(self):
        """Reuse the existing browse button for a bounded page of source IDs."""
        self.settings_open=False;self.inspection=('list',None);self._set_info(f'Simulated source neurons | page {self.neuron_page+1}/19\nSelect an ID to inspect its source morphology and evidence.')
        nodes=self.manifest['nodes'][self.neuron_page*32:(self.neuron_page+1)*32]
        rows=[(n['id']+' | '+', '.join(n['roles'][:2]),lambda id=n['id']:self.select(id)) for n in nodes]
        if self.neuron_page>0:rows.append(('Previous page',lambda:self._browse_page(-1)))
        if (self.neuron_page+1)*32<len(self.manifest['nodes']):rows.append(('Next page',lambda:self._browse_page(1)))
        self._buttons(rows)

    def _browse_page(self,delta):
        self.neuron_page+=delta;self.list_groups()

    def pick(self,x,y):
        if self.layout.route(x,y)!='brain' or not self._pick_parts:return
        if self._pick_points is None:
            self._pick_points=np.concatenate([points for _,points in self._pick_parts])
            self._pick_ids=np.array([id for id,points in self._pick_parts for _ in points])
        left,right,bottom,top=self.layout.rectangles['brain']
        ndc=np.array(((x-left)/(right-left)*2-1,(y-bottom)/(top-bottom)*2-1))
        local=np.c_[self._pick_points,np.ones(len(self._pick_points))]@np.array(self.brain_scene.getMat(self.brain_camera))
        fov=self.brain_camera.node().getLens().getFov()
        projected=local[:,[0,2]]/np.maximum(local[:,1:2],1e-9)/np.tan(np.radians(fov)/2)
        distances=np.linalg.norm(projected-ndc,axis=1)
        distances[(local[:,1]<=0)|((self._pick_points[:,1]<-.43)&(not self.vnc))]=np.inf
        index=int(distances.argmin())
        if distances[index]<.06:self.select(str(self._pick_ids[index]))

    def _update_activity(self,frame):
        token=(frame.episode_id,frame.step_index,self.vnc,self.detail_view,len(self.source_batches),self.activity_mask)
        if token==self._activity_token:return
        self.activity_table.update(frame,self.detail_view,self.activity_mask)
        owners=self._resident_owners[0] | (self._resident_owners[1] if self.vnc else set())
        strength=pulse_strengths(frame)
        self.visible_activity_ids=tuple(id for i,id in enumerate(frame.snapshot.neuron_ids) if i in owners and self.activity_mask[i] and strength[i]>0)
        self.enabled_firing=sum(on and count>0 for on,count in zip(self.activity_mask,frame.snapshot.spike_counts))
        self._activity_token=token;self.activity_updates+=1

    def draw_snapshot(self,snapshot):
        """Historical drawing entry point: use publication, never record a draw.

        Default-viewer callers must step through the pump. The argument cannot
        create display history or replace part of the atomic publication.
        """
        self.draw_frame(self.simulation.display_frame)

    def draw_frame(self,frame: DisplayFrame):
        """Consume one complete immutable frame without advancing any history."""
        self._display_frame=frame
        snapshot=frame.snapshot
        self._world_camera()
        token=(frame.episode_id,frame.step_index)
        if token!=self._body_token:
            self.fly.setPos(*snapshot.position);self.fly_marker.setPos(*snapshot.position)
            roll,pitch,yaw=snapshot.orientation
            self.fly.setHpr(-math.degrees(yaw),-math.degrees(pitch),math.degrees(roll))
            for sign,wing in zip((-1,1),self.wings):wing.setR(sign*35*math.sin(snapshot.time*math.tau*12)*snapshot.command.power)
            self.legs.setSz(.25+.75*snapshot.command.legs);self._body_token=token
        self._update_activity(frame)
        self._refresh_selected_info(frame)
        hud=(f'{snapshot.mode.upper()} | seed {snapshot.seed} | {snapshot.time:.2f}s | '
            f'power {snapshot.command.power:.2f} | contact {"food" if snapshot.food_contact else "none"} | dwell {snapshot.dwell:.2f}s'
            +(' | PAUSED' if self.paused else '')+(' | IMPACT' if snapshot.crashed else ''))
        if self.hud.getText()!=hud:self.hud.setText(hud)
        label='OBSERVATIONAL neuron activity | '+snapshot.mode+' supplies commands' if snapshot.mode in ('teacher','manual') else 'Modeled neuron activity | brightness does not prove importance'
        if self.activity_label.getText()!=label:self.activity_label.setText(label)
        self._refresh_coverage()

    def _world_camera(self):
        """Follow the same acquired snapshot as body, inspector and activity."""
        snapshot=self._display_frame.snapshot if self._display_frame else self.session.snapshot
        target=np.array(snapshot.position) if self.follow else np.array((.3,.3,.1))
        a,e=self.world_azimuth,self.world_elevation
        offset=self.world_distance*np.array((math.sin(a)*math.cos(e),-math.cos(a)*math.cos(e),math.sin(e)))
        self.camera.setPos(*(target+offset));self.camera.lookAt(Point3(*target),Vec3(0,0,1))
        if hasattr(self,'world_motion'):
            self.camera.setPos(self.camera.getPos()+Vec3(*self.world_motion.offset))

    def toggle_pause(self):
        self.paused=not self.paused
        if self.paused:self.simulation.pause()
        else:self.simulation.resume()

    def replay(self):
        self.simulation.reset(self.session.replay);self.accumulator=0.
        self._activity_token=None;self._body_token=None
        self.draw_snapshot(self.session.snapshot)

    def new_trial(self):
        self.simulation.reset(self.session.new_trial);self._flower();self.accumulator=0.
        self._activity_token=None;self._body_token=None
        self.draw_snapshot(self.session.snapshot)

    def destroy(self):
        if hasattr(self,'simulation'):self.simulation.close()
        super().destroy()

    def userExit(self):
        """Stop session mutation before the native window exits."""
        if hasattr(self,'simulation'):self.simulation.close()
        super().userExit()

    def tick(self,task):
        if self.ui_size!=(self.win.getXSize(),self.win.getYSize()):self._ui();self._resize()
        dt=min(ClockObject.getGlobalClock().getDt(),.1)
        if self.drag:
            if self.mouseWatcherNode and self.mouseWatcherNode.hasMouse():
                mouse=self.mouseWatcherNode.getMouse();self.move_gesture(mouse.x,mouse.y)
            else:self.cancel_gesture()
        self.brain_motion.advance(dt);self.world_motion.advance(dt)
        self.world_distance=self.world_motion.distance
        self._brain_camera()
        if not self.paused:
            self.simulation.resume()
            command=FlightCommand(.5+.2*(('arrow_up' in self.keys)-('arrow_down' in self.keys)),
                .4*(('w' in self.keys)-('s' in self.keys)),.4*(('d' in self.keys)-('a' in self.keys)),
                .3*(('e' in self.keys)-('q' in self.keys)),float('l' in self.keys))
            self.simulation.request(dt,command)
        else:self.simulation.pause()
        self.load_geometry();self.draw_frame(self.simulation.display_frame)
        return task.cont

    def capture(self,directory: Path):
        """Export opaque source-anatomy previews before and after activity emphasis."""
        directory.mkdir(parents=True,exist_ok=True);self.paused=True;self.simulation.pause();self.load_geometry(100000)
        def save(name):
            self.draw_frame(self.simulation.display_frame)
            for _ in range(3):self.taskMgr.step();self.graphicsEngine.syncFrame()
            picture=PNMImage();self.win.getScreenshot(picture);picture.removeAlpha();picture.write(str(directory/name))
        self.brain_motion=BrainCamera();self._brain_camera()
        for name in ('front','side','top'):self.preset(name);save(f'anatomy-{name}.png')
        self.simulation.advance_fixed(50)
        for name in ('front','side','top'):self.preset(name);save(f'activity-{name}.png')
        self.toggle_vnc();self.load_geometry(100000);self.preset('front');self.fit_view();self.brain_motion.settle();self._brain_camera();save('brain-and-cord.png')
        self.focus_brain();self.select(next(id for id in self.loaded if self.nodes[id]['metadata']['region']=='optic_lobe'))
        if self._selection_pending:self._selection_pending[1].result()
        self.load_geometry(100000);self.preset('front');save('source-selection.png')
