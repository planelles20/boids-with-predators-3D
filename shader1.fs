#version 330 core

in vec3 finalVel;
in vec3 finalColour;
flat in int  finalTag;

out vec4 color;

void main() {
    //color = vec4(abs(finalVel), 1.0f);
    if(finalTag > 0)
        color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    else
        color = vec4(1.0f, 1.0f, 1.0f, 1.0f);

}
