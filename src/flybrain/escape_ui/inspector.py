"""Readable source inspection; long identifiers and pathway data are opt-in."""
from .groups import GROUPS
from .widgets import Widgets
from flybrain.encounters3d.diagnostic_panel import pathway_text


class DashboardInspector:
    """Reuse source selection/loading while changing only its presentation."""
    def _group_for(self, id: str) -> str | None:
        owner = self._owner_index.get(id)
        return next((key for key, members in self.activity_index.members.items()
                     if owner in members), None)

    def _source_type(self, id: str) -> str:
        record = self.records.get(id, {})
        node = self.nodes.get(id, {})
        return record.get('type') or node.get('metadata', {}).get('cell_type') or 'Unspecified'

    def _selection_info(self, frame) -> str:
        id = self.selected
        key = self._group_for(id)
        name, description = GROUPS[key] if key else ('Source neuron', 'Not simulated in this subset.')
        lines = [name, f'Source cell type: {self._source_type(id)}', '', description]
        if self._selection_filtered():
            lines += ['', 'Hidden by filter. Tick its group to show it.']
        if frame and id in self._owner_index:
            i = self._owner_index[id]
            lines += ['', 'Recorded activity', f'Recent firing: {frame.firing_rates_hz[i]:.1f} spikes/s',
                      f'Latest interval: {frame.snapshot.spike_counts[i]} spikes',
                      f'Since this encounter: {frame.cumulative_counts[i]} spikes']
        if self.technical_open:
            incoming, outgoing, _ = self._edge_counts.get(id, (0, 0, 0))
            node = self.nodes.get(id, {})
            lines += ['', 'Technical details', f'Source ID: {id}',
                      f'Retained connections: {incoming} in / {outgoing} out',
                      f'Source: BANC v888 / {self.records[id]["region"]}',
                      f'Model roles: {", ".join(node.get("roles", []))}',
                      f'Evidence: {node.get("evidence", "source annotation")}']
            lines += ['', self._diagnostic_text(frame)]
        return '\n'.join(lines)

    def _diagnostic_text(self, frame) -> str:
        if frame is None or not hasattr(frame.snapshot, 'escape_state'):
            return 'No completed encounter interval yet.'
        s = frame.snapshot
        status = 'Verified declared edge protocol' if self.edge_verified else 'Development preview'
        return (f'{status}\nHistorical direction: 94/96; two known failures.\n'
            'Neural turn decision; engineered flight and hover.\nNo online training.\n'
            f'Seed {s.seed} / {s.time:.2f}s / escape {s.escape_state}\n\n'
            + pathway_text(frame, self.manifest).replace(
                'No escape decoder; spikes do not command retreat.',
                'Paired steering spikes select the turn side.'))

    def _set_info(self, value: str) -> None:
        self.info_text = value
        if hasattr(self, 'info'):
            self.info.setText(value)

    def _buttons(self, rows) -> None:
        for button in self.member_buttons:
            button.destroy()
        self.member_buttons = []
        widgets = Widgets(self.panels['details'].getCanvas(), self.member_buttons, self.ui_font)
        for label, command in rows:
            widgets.button(label, 8, 0, self.layout.sidebar.width-64,
                           command, height=38, size=16)
        self._position_inspector_buttons()

    def _position_inspector_buttons(self) -> None:
        bounds = self.info.getTightBounds()
        y = -float(bounds[0].z)+30 if bounds else 30
        for button in self.member_buttons:
            button.setZ(-y)
            y += 46
        self.panels['details']['canvasSize'] = (0, self.layout.sidebar.width-48, -y-20, 0)

    def _show_selection_info(self) -> None:
        if not hasattr(self, 'info') or not self.selected:
            return
        self._set_info(self._selection_info(self._display_frame))
        self._buttons([('Focus neuron', self.focus_selected), ('Clear selection', self.clear_selection),
            ('Hide technical details' if self.technical_open else 'Technical details', self.toggle_technical)])
        self.member_buttons[0]['state'] = 'disabled' if self._selection_filtered() else 'normal'
        self._sync_selection_visibility()
        self._selection_info_token = None

    def _restore_inspector(self) -> None:
        if not hasattr(self, 'info'):
            return
        if self.selected:
            self._show_selection_info()
        else:
            self.list_groups()

    def list_groups(self) -> None:
        """Browse readable type/side labels within the currently described group."""
        if not hasattr(self, 'info'):
            return
        self.settings_open = False
        self.inspection = ('list', None)
        members = sorted(self.activity_index.members[self.focused_group])
        total_pages = max(1, (len(members)+5)//6)
        self.browser_page = min(self.browser_page, total_pages-1)
        name = GROUPS[self.focused_group][0]
        value = (f'Inspect a neuron\nClick a branch in the brain view or choose a cell below.\n\n'
                 f'{name} / page {self.browser_page+1} of {total_pages}')
        if self.technical_open:
            value += '\n\n'+self._diagnostic_text(self._display_frame)
        self._set_info(value)
        rows = [('Hide technical details' if self.technical_open else 'Technical details', self.toggle_technical)]
        for ordinal, owner in enumerate(members[self.browser_page*6:(self.browser_page+1)*6],
                                        self.browser_page*6+1):
            id = self.activity_index.ids[owner]
            side = str(self.nodes[id].get('metadata', {}).get('side', '')).capitalize()
            label = f'{self._source_type(id)} {side} / cell {ordinal}'
            rows.append((label, lambda id=id: self.select(id)))
        if self.browser_page > 0:
            rows.append(('Previous cells', lambda: self._browser_step(-1)))
        if self.browser_page+1 < total_pages:
            rows.append(('Next cells', lambda: self._browser_step(1)))
        self._buttons(rows)

    def _browser_step(self, direction: int) -> None:
        self.browser_page += direction
        self.list_groups()

    def select(self, id: str) -> None:
        if id not in self.records:
            return
        self.technical_open = False
        self.sidebar_tab = 'inspect'
        super().select(id)
        self._sync_selection_visibility()
        self.show_tab('inspect')

    def toggle_technical(self) -> None:
        self.technical_open = not self.technical_open
        self._restore_inspector()

    def toggle_diagnostics(self) -> None:
        self.technical_open = True
        self.show_tab('inspect')

    def _refresh_selected_info(self, frame) -> None:
        if not self.selected or self.sidebar_tab != 'inspect':
            return
        token = (frame.episode_id, frame.step_index, self.selected,
                 self.technical_open, self.enabled_classes, self.vnc)
        if token == self._selection_info_token:
            return
        self._selection_info_token = token
        text = self._selection_info(frame)
        if text != self.info_text:
            scroll = self.panels['details'].verticalScroll['value']
            self._set_info(text)
            self._position_inspector_buttons()
            self.panels['details'].verticalScroll['value'] = scroll
