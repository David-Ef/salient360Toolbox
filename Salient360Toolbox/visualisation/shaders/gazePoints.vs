#version 130

in vec2  a_position;
in vec4  a_bg_color;
in float a_size;

out vec4  v_bg_color;
out float v_size;
out float v_linewidth;
out float v_antialias;

out vec2 UV;

void main (void) {
    v_bg_color = a_bg_color;
    v_size = a_size;
    v_linewidth = .1;
    v_antialias = .5;
    
    gl_Position = vec4(a_position, 1. ,1.);
    gl_PointSize = v_size + 2*(v_linewidth + 1.5*v_antialias);
}