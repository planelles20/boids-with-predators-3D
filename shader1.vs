#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 velocity;
layout (location = 2) in int tag;

out vec3 pass_vel;
out int pass_tag;

void main()
{
    gl_Position = vec4(2.0*position-1.0, 1.0f);
	pass_vel = velocity;
    pass_tag = tag;
}
