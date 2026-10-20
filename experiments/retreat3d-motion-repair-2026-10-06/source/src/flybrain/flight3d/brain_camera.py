"""Display-only quaternion camera. All distances here are millimetres.

The quaternion rotates a canonical right/up/back basis together. Unlike an
azimuth/elevation camera it has no pole or angle-dependent up-vector switch.
Input targets and rendered state are separate; no simulation module is imported.
"""
import math
import numpy as np


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of scalar-first unit quaternions."""
    return np.r_[a[0]*b[0]-a[1:]@b[1:],
                 a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])]


def rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector without constructing a matrix."""
    t = 2*np.cross(q[1:], vector)
    return vector + q[0]*t + np.cross(q[1:], t)


def interpolate(a: np.ndarray, b: np.ndarray, amount: float) -> np.ndarray:
    """Shortest-arc spherical interpolation, including small rotations."""
    dot = float(a@b)
    if dot < 0: b = -b; dot = -dot
    if dot > .999999:
        result = a + amount*(b-a)
    else:
        angle = math.acos(min(1., dot))
        result = (math.sin((1-amount)*angle)*a + math.sin(amount*angle)*b)/math.sin(angle)
    return result / np.linalg.norm(result)


class BrainCamera:
    """Pixel rotation, scale-aware pan and explicit smooth fit with no inertia."""
    def __init__(self) -> None:
        self.minimum = .03; self.maximum = 3.
        self.distance = self.target = 1.10
        self.pivot = np.array((.49, -.17, .145))
        self.pivot_target = self.pivot.copy()
        self.offset = np.zeros(3); self.offset_target = self.offset.copy()
        self.orientation = np.array((1., 0., 0., 0.))
        self.orientation_target = self.orientation.copy()

    @property
    def right(self) -> np.ndarray:
        return rotate(self.orientation, np.array((1., 0., 0.)))

    @property
    def up(self) -> np.ndarray:
        return rotate(self.orientation, np.array((0., 1., 0.)))

    @property
    def back(self) -> np.ndarray:
        return rotate(self.orientation, np.array((0., 0., 1.)))

    @property
    def centre(self) -> np.ndarray:
        return self.pivot + self.offset

    @property
    def position(self) -> np.ndarray:
        return self.centre + self.distance*self.back

    @property
    def near(self) -> float:
        """Adaptive clipping allows inspection at the 0.03 mm distance floor."""
        return max(.00001, min(.005, self.distance*.01))

    def preset(self, name: str) -> None:
        """Change orientation only; preserve the pivot, pan and zoom targets."""
        self.orientation = np.array({'front': (1., 0., 0., 0.),
            'side': (.5, .5, .5, .5),
            'top': (math.sqrt(.5), math.sqrt(.5), 0., 0.)}[name])
        self.orientation_target = self.orientation.copy()

    def rotate_pixels(self, dx: float, dy: float) -> None:
        """Apply 0.006 radians/pixel around the camera's own up/right axes."""
        vector = np.array((dy, -dx, 0.))*.006
        angle = float(np.linalg.norm(vector))
        if angle == 0: return
        local = np.r_[math.cos(angle/2), vector/angle*math.sin(angle/2)]
        self.orientation_target = multiply(self.orientation_target, local)
        self.orientation_target /= np.linalg.norm(self.orientation_target)

    def pan_pixels(self, dx: float, dy: float, height: float, vertical_fov: float) -> None:
        """Move at the current perspective scale without altering distance."""
        scale = 2*self.distance*math.tan(math.radians(vertical_fov)/2)/max(1., height)
        self.offset_target += scale*(-dx*self.right - dy*self.up)

    def zoom(self, factor: float) -> None:
        """Preserve wheel multipliers; pan never changes the distance floor."""
        self.target = float(np.clip(self.target*factor, self.minimum, self.maximum))

    def fit(self, bounds: np.ndarray, fov: tuple[float, float]) -> None:
        """Smoothly fit a visible bounding box with margin in both lens axes."""
        centre = bounds.mean(axis=0)
        corners = np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
        relative = corners-centre
        horizontal, vertical = np.tan(np.radians(fov)/2)
        depth = relative @ self.back
        required = np.maximum(abs(relative @ self.right)/horizontal,
                              abs(relative @ self.up)/vertical)*1.12 + depth
        self.pivot_target = centre
        self.offset_target = np.zeros(3)
        self.target = float(np.clip(required.max()+.005, self.minimum, self.maximum))

    def advance(self, dt: float) -> None:
        """Exponential responses: rotation/pan 60 ms; zoom/fit pivot 120 ms."""
        response = -math.expm1(-max(0., dt)/.06)
        zoom_response = -math.expm1(-max(0., dt)/.12)
        if not np.array_equal(self.orientation,self.orientation_target):
            self.orientation = interpolate(self.orientation, self.orientation_target, response)
        self.offset += (self.offset_target-self.offset)*response
        self.pivot += (self.pivot_target-self.pivot)*zoom_response
        self.distance += (self.target-self.distance)*zoom_response

    def release(self) -> None:
        """Clear pending gesture motion: releasing never leaves continued spin."""
        self.orientation_target = self.orientation.copy()
        self.offset_target = self.offset.copy()

    def settle(self) -> None:
        """Resolve explicit fit for startup/capture, never as a toggle side effect."""
        self.orientation = self.orientation_target.copy()
        self.pivot = self.pivot_target.copy(); self.offset = self.offset_target.copy()
        self.distance = self.target
