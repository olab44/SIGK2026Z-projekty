#version 330 core

uniform sampler2D tex;

in vec2 v_uv;
out vec4 f_color;

void main() {
    f_color = vec4(texture(tex, v_uv).rgb, 1.0);
}
