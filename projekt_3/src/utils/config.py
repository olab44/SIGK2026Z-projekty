from pathlib import Path

GL_VERSION = (3, 3)
WINDOW_TITLE = 'SIGK 4'
WINDOW_SIZE = (128, 128)

FRAGMENT_SHADER_EXTENSION = ['.frag']
VERTEX_SHADER_EXTENSION = ['.vert']


def get_supported_extensions():
    return [
        *VERTEX_SHADER_EXTENSION,
        *FRAGMENT_SHADER_EXTENSION
    ]


# Size of the reference image (rendered into an off-screen FBO).
IMAGE_SIZE = WINDOW_SIZE

# Camera stored in the provided program (do not change).
CAMERA_POSITION = (5.0, 5.0, 15.0)
CAMERA_TARGET = (0.0, 0.0, 0.0)
CAMERA_UP = (0.0, 1.0, 0.0)
FOV_DEG = 45.0
NEAR = 0.1
FAR = 1000.0
ASPECT = 1.0  # references are always projected with aspect 1 (not the window aspect)

# Scene.
SPHERE_RADIUS = 1.0
POS_RANGE = (-20.0, 20.0)       # object and light position, per component
DIFFUSE_RANGE = (0, 255)        # integers, both ends inclusive
SHININESS_RANGE = (3.0, 20.0)

# Dataset.
NUM_IMAGES = 3000
TEST_SIZE = 600
VAL_SIZE = 240
SEED = 42

VISIBILITY_MARGIN = 0.0         # sphere margin w.r.t. the frustum side planes (world units)
MIN_LIGHT_DISTANCE = 2.0        # min. distance between the light and the sphere centre
MAX_CAMERA_DISTANCE = None      # optional: max. distance from the camera to the sphere centre

# Constants hard-coded in phong.frag (documentation only; the shader is not modified).
# 0.3 * 255 = 76.5 ~ 76 and 0.1 * 255 = 25.5 ~ 25, i.e. the values from the assignment.
SHADER_CONSTANTS = {
    "material_ambient": 0.3,
    "material_specular": 1.0,
    "light_ambient": 0.1,
    "light_diffuse": 1.0,
    "light_specular": 1.0,
}

# Default paths (independent of the working directory).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'phong'
RUNS_DIR = PROJECT_ROOT / 'runs'
FIGURES_DIR = PROJECT_ROOT / 'figures'
