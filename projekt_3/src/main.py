from collections import namedtuple
from enum import Enum

import moderngl_window

import sys  # [SIGK-P3]

from phong_window import PhongWindow
from neural_window import NeuralWindow  # [SIGK-P3]

Task = namedtuple('Task', ['window_args', 'window_cls'])


class TaskType(Enum):
    @property
    def window_args(self):
        return self.value.window_args

    @property
    def window_cls(self):
        return self.value.window_cls

    PHONG = Task(
        [
            "--shaders_dir_path=../resources/shaders/phong",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            "--output_path=../data/phong/",  # [SIGK-P3] was ../output/
            "--num_images=3000",  # [SIGK-P3]
            "--seed=42",  # [SIGK-P3]
        ],
        PhongWindow
    )

    # [SIGK-P3] live comparison: renderer (left half) | network (right half)
    NEURAL = Task(
        [
            "--shaders_dir_path=../resources/shaders/phong",
            "--shader_name=phong",
            "--model_name=sphere.obj",
        ],
        NeuralWindow
    )


if __name__ == '__main__':
    # [SIGK-P3] an optional first argument selects the task (phong | neural); remaining arguments
    # (e.g. --window headless, --num_images 10, --checkpoint ...) override the defaults.
    argv = sys.argv[1:]
    task = TaskType.PHONG
    if argv and argv[0].upper() in TaskType.__members__:
        task = TaskType[argv[0].upper()]
        argv = argv[1:]
    moderngl_window.run_window_config(task.window_cls, args=task.window_args + argv)
