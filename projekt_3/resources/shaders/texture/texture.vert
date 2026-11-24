#version 330 core

// Full-screen quad (attributes from geometry.quad_2d): in_position, in_texcoord_0.
in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 v_uv;

void main() {
    v_uv = in_texcoord_0;
    gl_Position = vec4(in_position, 1.0);
}
