#version 130

in vec4 v_bg_color;
in float v_size;
in float v_linewidth;
in float v_antialias;
//in vec2 UV;

out vec4 color;

float disc(vec2 P, float size)
{
    float r = length((P.xy - vec2(0.5,0.5))*size);
    r -= v_size/2;
    return r;
}
void main()
{
    vec4 v_fg_color = vec4(.1);
  
    // color = v_bg_color;
    // color.w = 1;
    // color.x = gl_PointCoord.x;
    // color.y = gl_PointCoord.y;
    // return;

    float size = v_size +2*(v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;

    float r = disc(gl_PointCoord, size);
    float d = abs(r) - t;
    if( r > (v_linewidth/2.0+v_antialias))
    {
        discard;
    }
    else if(d < 0.0 )
    {
       color = v_fg_color;
    }
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > 0){
            color = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        }
        else{
            color = mix(v_bg_color, v_fg_color, alpha);
        }
    }
}