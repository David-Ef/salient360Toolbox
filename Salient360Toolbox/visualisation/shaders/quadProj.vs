#version 130

in vec3 vert;
in vec2 uV;

uniform mat4 mvMatrix;

out vec3 Normal;
out vec2 UV;

void main() {
  gl_Position = mvMatrix * vec4(vert, 1.0);
  UV = uV;
}