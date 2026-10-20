"""One display mask governs branches, selection, picking and camera bounds."""
import numpy as np
import torch
from flybrain.flight3d.viewer import geometry
from .groups import GroupActivityTable


class GroupFilters:
    """Presentation mixin; never advances, resets or disables a simulated cell."""
    def _anatomy(self) -> None:
        super()._anatomy()
        self.activity_table = GroupActivityTable(self.session.brain.ids,
            np.asarray(self.activity_index.activation_colors))

    def _append_source(self, lines, cord: bool, owner: int) -> None:
        if len(lines):
            points = lines.reshape(-1, 3)
            self.group_bounds[(owner, bool(cord))] = np.array((points.min(axis=0), points.max(axis=0)))
        super()._append_source(lines, cord, owner)

    def _prepare_selection(self, lines) -> None:
        """Remember full highlighted morphology in addition to coarse branches."""
        super()._prepare_selection(lines)
        self._selection_bound_owner = self.selected
        self.selection_bounds = {}
        regions = lines.mean(axis=1)[:, 1] < -.43
        for cord in (False, True):
            points = lines[regions == cord].reshape(-1, 3)
            if len(points):
                self.selection_bounds[cord] = np.array((points.min(axis=0), points.max(axis=0)))

    def set_group_visible(self, key: str, enabled: bool) -> None:
        """Apply explicit checked state, including repeated identical callbacks."""
        if key not in self.activity_index.members:
            raise ValueError('unknown neuron group')
        self.focused_group = key
        if enabled:
            self.enabled_classes = self.enabled_classes | frozenset({key})
        else:
            self.enabled_classes = self.enabled_classes - frozenset({key})
        self._refresh_activity_choices()

    def show_only_group(self, key: str) -> None:
        if key not in self.activity_index.members:
            raise ValueError('unknown neuron group')
        self.focused_group = key
        self.enabled_classes = frozenset({key})
        self.activity_scope = 'whole'
        self._refresh_activity_choices()

    def show_all_groups(self) -> None:
        self.enabled_classes = frozenset(self.activity_index.members)
        self.activity_scope = 'whole'
        self._refresh_activity_choices()

    def hide_all_groups(self) -> None:
        self.enabled_classes = frozenset()
        self._refresh_activity_choices()

    def all_activity(self) -> None:
        self.show_all_groups()

    def activity_off(self) -> None:
        self.hide_all_groups()

    def _refresh_activity_choices(self) -> None:
        super()._refresh_activity_choices()
        self._pick_points = None
        self._sync_selection_visibility()
        self._draw_connections()
        self._sync_group_controls()
        self._selection_info_token = None
        if self.selected:
            self._show_selection_info()

    def set_outline(self, enabled: bool) -> None:
        self.whole = bool(enabled)
        self._visibility()

    def toggle_mode(self) -> None:
        self.set_outline(not self.whole)

    def set_cord(self, enabled: bool) -> None:
        if self.vnc != bool(enabled):
            super().toggle_vnc()
        self._sync_group_controls()

    def toggle_vnc(self) -> None:
        self.set_cord(not self.vnc)

    def _visibility(self) -> None:
        super()._visibility()
        # Outline is the registered soft surface, not the dense reference-point
        # cloud that makes functional groups difficult to see.
        for context in self.context:
            context.find('**/registered-source-context').hide()
        self._sync_selection_visibility()
        self._sync_group_controls()

    def _selection_filtered(self) -> bool:
        owner = self._owner_index.get(self.selected)
        return owner is None or not self.activity_mask[owner]

    def _sync_selection_visibility(self) -> None:
        if not hasattr(self, 'selection_nodes'):
            return
        for node in self.selection_nodes:
            node.hide() if self.selected and self._selection_filtered() else node.show()

    def _draw_connections(self) -> None:
        if not hasattr(self, 'edge_node'):
            return
        self.edge_node.getChildren().detach()
        if not self.selected or not self.show_edges or self._selection_filtered():
            return
        lines = []
        for edge in self.manifest['edges']:
            ids = (edge['pre'], edge['post'])
            if self.selected not in ids or not all(id in self.root_positions for id in ids):
                continue
            if not all(self.activity_mask[self._owner_index[id]] for id in ids):
                continue
            if not self.vnc and any(self.root_positions[id][1] < -.43 for id in ids):
                continue
            lines.append(tuple(self.root_positions[id] for id in ids))
        if lines:
            geometry('visible-selected-connections', lines, kind='lines',
                     color=(.4, .8, .86, .5)).reparentTo(self.edge_node)

    def visible_bounds(self):
        """Bounds of visible source geometry, including the optional outline."""
        if not any(self.activity_mask):
            return None
        rows = []
        for (owner, cord), bounds in self.group_bounds.items():
            if self.activity_mask[owner] and (self.vnc or not cord):
                rows.extend(bounds)
        if (self.selected and not self._selection_filtered()
                and getattr(self, '_selection_bound_owner', None) == self.selected):
            for cord, bounds in self.selection_bounds.items():
                if self.vnc or not cord:
                    rows.extend(bounds)
        if self.whole:
            rows.extend(self._context_bounds[0])
            if self.vnc:
                rows.extend(self._context_bounds[1])
        if not rows:
            return None
        points = np.asarray(rows)
        return np.array((points.min(axis=0), points.max(axis=0)))

    def fit_view(self) -> None:
        bounds = self.visible_bounds()
        if bounds is not None:
            self.brain_motion.fit(bounds, tuple(self.brain_camera.node().getLens().getFov()))

    def focus_selected(self) -> None:
        if self.selected and not self._selection_filtered():
            super().focus_selected()

    def pick(self, x: float, y: float) -> None:
        """Project only enabled owners; filtered branches cannot win a pick."""
        if self.layout.route(x, y) != 'brain' or not self._pick_parts:
            return
        if self._pick_points is None:
            self._pick_points = np.concatenate([points for _, points in self._pick_parts])
            self._pick_ids = np.array([id for id, points in self._pick_parts for _ in points])
        if not len(self._pick_points):
            return
        left, right, bottom, top = self.layout.rectangles['brain']
        ndc = np.array(((x-left)/(right-left)*2-1, (y-bottom)/(top-bottom)*2-1))
        local = np.c_[self._pick_points, np.ones(len(self._pick_points))] @ np.array(self.brain_scene.getMat(self.brain_camera))
        fov = self.brain_camera.node().getLens().getFov()
        projected = local[:, [0, 2]] / np.maximum(local[:, 1:2], 1e-9) / np.tan(np.radians(fov)/2)
        distances = np.linalg.norm(projected-ndc, axis=1)
        visible = np.array([self.activity_mask[self._owner_index[id]] for id in self._pick_ids])
        distances[(local[:, 1] <= 0) | ~visible | ((self._pick_points[:, 1] < -.43) & (not self.vnc))] = np.inf
        index = int(distances.argmin())
        if distances[index] < .06:
            self.select(str(self._pick_ids[index]))
