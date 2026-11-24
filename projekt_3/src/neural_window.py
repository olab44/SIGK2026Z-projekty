"""[SIGK-P3] Network wired to the renderer: left = Phong renderer, right = generator output.

Inherits from PhongWindow to reuse the sphere mesh, the Phong program and `render_reference`.
Controls: SPACE - sample a new scene, S - save the pair of images to ../output/neural/.
"""
from pathlib import Path

import moderngl
import numpy as np
from moderngl_window import geometry
from PIL import Image

from phong_window import PhongWindow
from scene_sampling import sample_scene
from utils import config as cfg
from utils.shader_utils import get_shaders

TEXTURE_SHADERS_DIR = (Path(__file__).parent.parent / 'resources' / 'shaders' / 'texture').resolve()


class NeuralWindow(PhongWindow):
    window_size = (2 * cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1])  # left half: renderer, right half: network
    aspect_ratio = None  # fill the whole window (the reference projection uses cfg.ASPECT anyway)
    title = 'SIGK 3 - renderer | neural network'

    @classmethod
    def add_arguments(cls, parser):
        super().add_arguments(parser)
        parser.add_argument('--checkpoint', type=str, required=True, help='Path to best.pt / last.pt')
        parser.add_argument('--encoding', type=str, choices=['raw', 'relative'], default=None,
                            help='Encoding of the conditioning vector (default: from the checkpoint)')
        parser.add_argument('--device', type=str, default='auto')
        parser.add_argument('--save_dir', type=str, default='../output/neural/')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Import torch only here so that data generation does not require PyTorch.
        from neural.models import load_generator
        from neural.train import pick_device

        self.device = pick_device(self.argv.device)
        self.G, model_cfg = load_generator(self.argv.checkpoint, self.device)
        self.encoding = self.argv.encoding or model_cfg['encoding']
        if self.encoding != model_cfg['encoding']:
            print(f"WARNING: encoding {self.encoding} differs from the training encoding ({model_cfg['encoding']})")
        self.save_dir = Path(self.argv.save_dir)
        self.saved = 0
        self.dirty = True
        self.ref_rgb = self.gen_rgb = None
        self.scene = None

        shaders = get_shaders(str(TEXTURE_SHADERS_DIR))['texture']
        self.tex_program = self.ctx.program(vertex_shader=shaders.vertex_shader,
                                            fragment_shader=shaders.fragment_shader)
        self.tex_program['tex'].value = 0
        # quad_2d attribute names: in_position, in_texcoord_0 (moderngl_window.geometry.attributes)
        self.quad_left = geometry.quad_2d(size=(1.0, 2.0), pos=(-0.5, 0.0), normals=False).instance(self.tex_program)
        self.quad_right = geometry.quad_2d(size=(1.0, 2.0), pos=(0.5, 0.0), normals=False).instance(self.tex_program)
        w, h = cfg.IMAGE_SIZE
        self.tex_ref = self.ctx.texture((w, h), 3)
        self.tex_gen = self.ctx.texture((w, h), 3)
        for t in (self.tex_ref, self.tex_gen):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)

    # ------------------------------------------------------------------ logic
    def new_scene(self):
        """Sample a scene, render the reference and generate the network image."""
        import torch
        from neural.encoding import encode, row_from_scene
        from neural.train import generate, to_uint8

        self.scene = sample_scene(self.rng, self.rejection_stats)
        self.ref_rgb = self.render_reference(self.scene)[..., :3].copy()
        cond = torch.from_numpy(encode(row_from_scene(self.scene), self.encoding))[None].to(self.device)
        self.gen_rgb = to_uint8(generate(self.G, cond))[0]
        # GL textures have row 0 at the bottom while numpy images have it at the top -> flip.
        self.tex_ref.write(np.ascontiguousarray(self.ref_rgb[::-1]).tobytes())
        self.tex_gen.write(np.ascontiguousarray(self.gen_rgb[::-1]).tobytes())
        self.dirty = False

    def save_pair(self):
        """Save the renderer image and the network image of the current scene."""
        if self.ref_rgb is None:
            return None
        self.save_dir.mkdir(parents=True, exist_ok=True)
        stem = self.save_dir / f'pair_{self.saved:03}'
        Image.fromarray(self.ref_rgb).save(f'{stem}_renderer.png')
        Image.fromarray(self.gen_rgb).save(f'{stem}_neural.png')
        print(f'saved {stem}_*.png')
        self.saved += 1
        return stem

    # ------------------------------------------------------------------ window
    def on_render(self, time: float, frame_time: float):
        if self.dirty:
            self.new_scene()
        self.wnd.use()
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.tex_ref.use(location=0)
        self.quad_left.render()
        self.tex_gen.use(location=0)
        self.quad_right.render()

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        if key == keys.SPACE:
            self.dirty = True
        elif key == keys.S:
            self.save_pair()
