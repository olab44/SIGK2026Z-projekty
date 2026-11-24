import csv  # [SIGK-P3]
import json  # [SIGK-P3]
import os.path
import platform  # [SIGK-P3]
import sys  # [SIGK-P3]
from collections import Counter  # [SIGK-P3]

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

from base_window import BaseWindow
from scene_sampling import sample_scene  # [SIGK-P3]
from utils import config as cfg  # [SIGK-P3]

# [SIGK-P3] Columns of params.csv (raw world parameters; encoding is done by the training code).
PARAM_COLUMNS = ['frame', 'obj_x', 'obj_y', 'obj_z', 'light_x', 'light_y', 'light_z',
                 'kd_r', 'kd_g', 'kd_b', 'shininess', 'attempts']


class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0
        # [SIGK-P3] data generator state
        self.num_images = self.argv.num_images
        self.seed = self.argv.seed
        self.rng = np.random.default_rng(self.seed)
        self.rejection_stats = Counter()
        self.param_rows = []
        self.finished = False
        self._create_offscreen_fbo()

    # [SIGK-P3] extra CLI arguments (calls super)
    @classmethod
    def add_arguments(cls, parser):
        super().add_arguments(parser)
        parser.add_argument('--num_images', type=int, default=cfg.NUM_IMAGES,
                            help='Number of images to generate (then the window closes)')
        parser.add_argument('--seed', type=int, default=cfg.SEED, help='Seed of the scene sampler')

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    # [SIGK-P3] Off-screen 128x128 FBO (RGBA + depth renderbuffer, no MSAA): the result does not
    # depend on the screen DPI or on the window size.
    def _create_offscreen_fbo(self):
        self.image_size = tuple(cfg.IMAGE_SIZE)
        self.offscreen_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture(self.image_size, 4)],
            depth_attachment=self.ctx.depth_renderbuffer(self.image_size),
        )

    # [SIGK-P3] Extracted from on_render: renders a scene (dict from sample_scene) into the FBO.
    def render_reference(self, scene) -> np.ndarray:
        """Render a scene and return RGBA uint8 of shape (H, W, 4), row 0 = top of the image.

        The alpha channel is the sphere mask (the shader writes alpha 1.0, background is cleared to 0).
        """
        self.offscreen_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        model_translation = [float(c) for c in scene['obj']]
        material_diffuse = [float(c) / 255.0 for c in scene['diffuse']]  # [SIGK-P3] /255
        material_shininess = float(scene['shininess'])
        light_position = [float(c) for c in scene['light']]

        camera_position = list(cfg.CAMERA_POSITION)
        model_matrix = Matrix44.from_translation(model_translation)
        # [SIGK-P3] constants from config.py; aspect is always ASPECT (not the window aspect)
        proj = Matrix44.perspective_projection(cfg.FOV_DEG, cfg.ASPECT, cfg.NEAR, cfg.FAR)
        lookat = Matrix44.look_at(
            camera_position,
            cfg.CAMERA_TARGET,
            cfg.CAMERA_UP,
        )

        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

        self.vao.render()

        w, h = self.image_size
        raw = self.offscreen_fbo.read(components=4)
        # top-bottom flip as in the original (FLIP_TOP_BOTTOM): GL reads from the bottom
        image = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)[::-1].copy()

        self.wnd.use()  # restore the window framebuffer
        return image

    # [SIGK-P3] show the last render in the window (secondary; errors are ignored)
    def _blit_to_window(self):
        try:
            self.ctx.copy_framebuffer(self.wnd.fbo, self.offscreen_fbo)
        except Exception:  # e.g. different sizes when DPI > 1
            pass

    def on_render(self, time: float, frame_time: float):
        # [SIGK-P3] values are sampled here (instructor's hint); stop after N images
        if self.frame >= self.num_images:
            self.finish()
            return

        scene = sample_scene(self.rng, self.rejection_stats)
        image = self.render_reference(scene)
        self._blit_to_window()

        if self.output_path:
            images_dir = os.path.join(self.output_path, 'images')
            os.makedirs(images_dir, exist_ok=True)
            Image.fromarray(image, 'RGBA').save(os.path.join(images_dir, f'image_{self.frame:04}.png'))
            o, l, d = scene['obj'], scene['light'], scene['diffuse']
            self.param_rows.append([self.frame, *[f'{c:.6f}' for c in o], *[f'{c:.6f}' for c in l],
                                    *[int(c) for c in d], f"{scene['shininess']:.6f}", scene['attempts']])
        self.frame += 1

    # [SIGK-P3] write metadata and close the window
    def finish(self):
        if self.finished:
            return
        self.finished = True
        if self.output_path:
            with open(os.path.join(self.output_path, 'params.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(PARAM_COLUMNS)
                writer.writerows(self.param_rows)
            with open(os.path.join(self.output_path, 'dataset_meta.json'), 'w') as f:
                json.dump(self._build_meta(), f, indent=2, ensure_ascii=False)
        self.wnd.close()

    def _build_meta(self):
        import moderngl_window
        import pyrr
        stats = dict(self.rejection_stats)
        attempts = stats.get('attempts', 0)
        accepted = stats.get('accepted', 0)
        constants = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()
                     if k.isupper() and not k.startswith('_') and not isinstance(v, (os.PathLike, list))
                     and k not in ('FRAGMENT_SHADER_EXTENSION', 'VERTEX_SHADER_EXTENSION')}
        return {
            'num_images': self.frame,
            'seed': self.seed,
            'constants': constants,
            'total_attempts': attempts,
            'accepted': accepted,
            'acceptance_rate': (accepted / attempts) if attempts else None,
            'rejections': {k: v for k, v in stats.items() if k not in ('attempts', 'accepted')},
            'packages': {
                'python': sys.version.split()[0],
                'platform': platform.platform(),
                'numpy': np.__version__,
                'moderngl': moderngl.__version__,
                'moderngl_window': moderngl_window.__version__,
                'pyrr': getattr(pyrr, '__version__', 'unknown'),
                'pillow': __import__('PIL').__version__,
            },
            'gl_renderer': self.ctx.info.get('GL_RENDERER'),
        }
