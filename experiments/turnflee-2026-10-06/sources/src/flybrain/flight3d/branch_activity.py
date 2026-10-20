"""Static owned branches sample a small whole-neuron RGBA activity table."""
import numpy as np
import torch  # Initialize Torch DLLs before Panda3D on Windows.
from panda3d.core import (Geom, GeomNode, GeomLines, GeomVertexData,
                         GeomVertexArrayFormat, GeomVertexFormat, InternalName,
                         NodePath, Shader, Texture, SamplerState, TransparencyAttrib)
from .activity import DisplayFrame, pulse_strengths

VERTEX_SHADER = '''#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform sampler2D activity_colors;
uniform float activity_width;
in vec4 p3d_Vertex;
in float activity_index;
out vec4 neuron_color;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    neuron_color = texture(activity_colors, vec2((activity_index + 0.5) / activity_width, 0.5));
}
'''
FRAGMENT_SHADER = '''#version 130
in vec4 neuron_color;
out vec4 fragment_color;
void main() { fragment_color = neuron_color; }
'''


class BranchActivityTable:
    """One texel per modeled neuron; geometry and neuron rules stay fixed."""
    def __init__(self, ids: tuple[str, ...], activation_colors: np.ndarray | None = None):
        self.ids = ids
        self.activation_colors = None
        if activation_colors is not None:
            palette = np.asarray(activation_colors, dtype=float)
            if palette.shape != (len(ids),3) or not np.isfinite(palette).all() or ((palette<0)|(palette>1)).any():
                raise ValueError('one finite unit-interval RGB activation color per neuron required')
            self.activation_colors = np.frombuffer(palette.tobytes(), dtype=float).reshape(-1,3)
        self.texture = Texture('whole-neuron-activity')
        self.texture.setup2dTexture(max(1,len(ids)),1,Texture.TUnsignedByte,Texture.FRgba8)
        self.texture.setMinfilter(SamplerState.FTNearest)
        self.texture.setMagfilter(SamplerState.FTNearest)
        self.texture.setWrapU(SamplerState.WMClamp)
        self.texture.setWrapV(SamplerState.WMClamp)
        self.shader = Shader.make(Shader.SLGLSL,VERTEX_SHADER,FRAGMENT_SHADER)
        self.colors = np.empty((len(ids),4))
        self.uploads = 0
        self._token = None
        self.upload_colors(np.tile((.24,.30,.55,.22),(len(ids),1)))
        self.uploads = 0

    def upload_colors(self, colors: np.ndarray) -> None:
        """Upload explicit RGBA bytes, avoiding Panda's native channel ordering."""
        colors = np.asarray(colors,dtype=float)
        if colors.shape != (len(self.ids),4) or not np.isfinite(colors).all():
            raise ValueError('finite per-neuron RGBA colors required')
        self.colors = np.frombuffer(np.clip(colors,0,1).tobytes(),dtype=float).reshape(-1,4)
        rgba = np.rint(self.colors*255).astype(np.uint8)
        self.texture.setRamImageAs(rgba.tobytes() if len(rgba) else bytes(4),'RGBA')
        self.uploads += 1
        self._token = None

    def update(self, frame: DisplayFrame, bright: bool, mask: tuple[bool, ...] | None = None) -> None:
        """Color firing owners in a display mask, retaining complete frame history.

        Omitted palette/mask preserve the original uncapped warm rendering.
        """
        if frame.snapshot.neuron_ids != self.ids:
            raise ValueError('display frame neuron order differs from branch owners')
        if mask is not None:
            mask=tuple(mask)
            if len(mask)!=len(self.ids) or any(not isinstance(v,(bool,np.bool_)) for v in mask):
                raise ValueError('one boolean display mask entry per neuron required')
        token=(frame.episode_id,frame.step_index,bright,mask)
        if token == self._token:return
        strength=pulse_strengths(frame)
        if mask is not None: strength=strength*np.asarray(mask)
        idle=np.array((.32,.40,.67,.42) if bright else (.24,.30,.55,.22))
        warm=np.column_stack((np.ones(len(strength)),.36+.48*strength,
                              np.full(len(strength),.07),.18+.25*strength))
        active=warm if self.activation_colors is None else np.column_stack((self.activation_colors,np.full(len(strength),.43)))
        self.upload_colors(idle+(active-idle)*strength[:,None])
        self._token=token

    def make_batch(self, segments: np.ndarray, owner_indices: np.ndarray, name: str) -> NodePath:
        """Build immutable endpoint positions and their matching neuron indices."""
        segments=np.asarray(segments,dtype=np.float32)
        owners=np.asarray(owner_indices)
        if segments.ndim!=3 or segments.shape[1:]!=(2,3) or owners.shape!=(len(segments),):
            raise ValueError('one owner per XYZ segment required')
        if not np.issubdtype(owners.dtype,np.integer) or ((owners<0)|(owners>=len(self.ids))).any():
            raise ValueError('valid integer neuron owner indices required')
        array=GeomVertexArrayFormat()
        array.addColumn(InternalName.getVertex(),3,Geom.NTFloat32,Geom.CPoint)
        array.addColumn(InternalName.make('activity_index'),1,Geom.NTFloat32,Geom.COther)
        vertex_format=GeomVertexFormat();vertex_format.addArray(array)
        vertex_format=GeomVertexFormat.registerFormat(vertex_format)
        data=GeomVertexData(name,vertex_format,Geom.UHStatic)
        data.setNumRows(len(segments)*2)
        vertices=np.empty((len(segments)*2,4),dtype=np.float32)
        vertices[:,:3]=segments.reshape(-1,3);vertices[:,3]=np.repeat(owners,2)
        data.modifyArray(0).modifyHandle().setData(vertices.tobytes())
        primitive=GeomLines(Geom.UHStatic)
        primitive.addConsecutiveVertices(0,len(vertices));primitive.closePrimitive()
        geom=Geom(data);geom.addPrimitive(primitive)
        node=GeomNode(name);node.addGeom(geom);path=NodePath(node)
        path.setShader(self.shader);path.setShaderInput('activity_colors',self.texture)
        path.setShaderInput('activity_width',float(max(1,len(self.ids))))
        path.setTransparency(TransparencyAttrib.MAlpha);path.setLightOff();path.setTwoSided(True)
        return path
