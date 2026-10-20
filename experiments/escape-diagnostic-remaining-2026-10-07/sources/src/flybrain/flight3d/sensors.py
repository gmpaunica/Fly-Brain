"""Deterministic ray-sampled vision and odor; no display dependency."""

import math
import numpy as np

from .contracts import FLOWER_RADIUS, NECTAR_RADIUS, FlightObservation

# Two 180-degree eyes tile the complete azimuth. Positive elevation is up.
AZIMUTH = np.stack((np.linspace(-math.pi,0,48,endpoint=False)+math.pi/96,
                    np.linspace(0,math.pi,48,endpoint=False)+math.pi/96))
ELEVATION = np.linspace(math.radians(80),math.radians(-80),24)
az = np.broadcast_to(AZIMUTH[:,None,:],(2,24,48))
el = np.broadcast_to(ELEVATION[None,:,None],(2,24,48))
EYE_RAYS = np.stack((np.sin(az)*np.cos(el),np.cos(az)*np.cos(el),np.sin(el)),axis=-1)
EYE_RAYS.flags.writeable = False

# Each pixel integrates eight independently occluded rays. Keep the original
# pixel centres and immutable [eye,row,column,RGB] observation contract.
_sub_az = az[...,None] + np.tile(np.array([-.25,.25]),4)*math.pi/48
_sub_el = el[...,None] + np.repeat(np.array([-.375,-.125,.125,.375]),2)*math.radians(160)/23
SUBPIXEL_RAYS = np.stack((np.sin(_sub_az)*np.cos(_sub_el),
    np.cos(_sub_az)*np.cos(_sub_el),np.sin(_sub_el)),axis=-1)
SUBPIXEL_RAYS.flags.writeable = False
CONTRAST_THRESHOLD = .01


def flower_contrast(rgb: np.ndarray) -> np.ndarray:
    """Positive flower-color contrast from RGB only, shared by both controllers.

    Reject contrast below .01; retain its magnitude above threshold. Sky,
    checker floor and green stem have negative contrast in this model.
    """
    values = np.asarray(rgb)
    contrast = values[...,0] - (values[...,1] + values[...,2]) * .5
    return np.where(contrast > CONTRAST_THRESHOLD, contrast, 0.)


def body_basis(orientation: np.ndarray) -> np.ndarray:
    """Columns are body right/forward/up in world XYZ. Angles: roll,pitch,yaw."""
    roll,pitch,yaw = orientation
    right = np.array((math.cos(yaw),-math.sin(yaw),0.))
    forward = np.array((math.sin(yaw)*math.cos(pitch),math.cos(yaw)*math.cos(pitch),-math.sin(pitch)))
    up = np.cross(right,forward)
    return np.column_stack((right*math.cos(roll)-up*math.sin(roll),forward,
                            up*math.cos(roll)+right*math.sin(roll)))


def eye_images(position: np.ndarray, orientation: np.ndarray, flower: tuple) -> np.ndarray:
    """RGB of sky, textured floor, stem and flower surfaces using nearest hits.

    Ray intersections belong to the environment. No hit distances, normals or
    object IDs cross the sensory boundary. Petals form a thin oblate ellipsoid.
    """
    rays = SUBPIXEL_RAYS @ body_basis(orientation).T
    shape = rays.shape[:-1]
    rgb = np.empty((*shape,3),dtype=np.float32); rgb[:] = (.035,.055,.09)
    nearest = np.full(shape,np.inf)
    dz = rays[...,2]
    ground_t = np.divide(-position[2],dz,out=np.full(shape,np.inf),where=dz < -1e-9)
    ground = ground_t>0
    ground &= np.isfinite(ground_t)
    safe_t = np.where(ground,ground_t,0.)
    gx=position[0]+rays[...,0]*safe_t; gy=position[1]+rays[...,1]*safe_t
    checker=(np.floor(gx/.025)+np.floor(gy/.025)).astype(np.int64)%2
    rgb[ground] = np.stack((.11+.025*checker,.17+.03*checker,.085+.02*checker),axis=-1)[ground]
    nearest[ground]=ground_t[ground]
    centre = np.array(flower)-np.array((0.,0.,.004))
    scale = np.array((FLOWER_RADIUS,FLOWER_RADIUS,.004))
    offset=(position-centre)/scale; rd=rays/scale
    a=np.sum(rd*rd,axis=-1); b=np.sum(rd*offset,axis=-1)
    disc=b*b-a*(np.dot(offset,offset)-1)
    t=(-b-np.sqrt(np.maximum(disc,0)))/a
    petal=(disc>=0)&(t>0)&(t<nearest)
    rgb[petal]=(.82,.12,.48); nearest[petal]=t[petal]
    # Horizontal nectar disc, visible from above; supports physical feet.
    t=np.divide(flower[2]-position[2],dz,out=np.full(shape,np.inf),where=np.abs(dz)>1e-9)
    ts=np.where(np.isfinite(t),t,0)
    dx=position[0]+rays[...,0]*ts-flower[0]; dy=position[1]+rays[...,1]*ts-flower[1]
    nectar=(t>0)&(t<nearest)&(dx*dx+dy*dy<=NECTAR_RADIUS**2)&(dz<0)
    rgb[nectar]=(.98,.69,.06); nearest[nectar]=t[nectar]
    # Stem cylinder occludes other objects, including when looking upward.
    ox=position[0]-flower[0]; oy=position[1]-flower[1]
    a=rays[...,0]**2+rays[...,1]**2
    b=ox*rays[...,0]+oy*rays[...,1]
    disc=b*b-a*(ox*ox+oy*oy-.002**2)
    t=np.divide(-b-np.sqrt(np.maximum(disc,0)),a,out=np.full(shape,np.inf),where=a>1e-12)
    z=position[2]+rays[...,2]*np.where(np.isfinite(t),t,0)
    stem=(disc>=0)&(t>0)&(t<nearest)&(z>=0)&(z<flower[2]-.004)
    rgb[stem]=(.09,.38,.12)
    return rgb.mean(axis=-2)


def observe_world(world) -> FlightObservation:
    """Sample local simulated sensors; this function alone may inspect world geometry."""
    basis=body_basis(world.orientation)
    antenna=world.position+basis[:,1]*.002
    flower=np.array(world.case.flower)
    odor=[]
    for side in (-1,1):
        delta=antenna+side*.001*basis[:,0]-flower
        odor.append(.0025/(float(np.dot(delta,delta))+.0025))
    return FlightObservation(*odor,eye_images(world.position,world.orientation,world.case.flower),
        tuple(world.angular_velocity),tuple(world.specific_force),world.foot_contact,world.food_contact)
