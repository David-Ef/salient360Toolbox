#version 130
#define PI 3.1415926535897932384626433832795

in vec2 UV;
uniform sampler2D backgroundTexture;
uniform sampler2D VPTexture;
uniform vec3 camAngle;
uniform vec2 angleSpan;

out vec3 colour;

mat3 getRotMat(vec3 angleCam){

	mat3 rotMat;

	float cosA = cos(angleCam.x);
	float sinA = sin(angleCam.x);
	float cosE = cos(angleCam.y);
	float sinE = sin(angleCam.y);
	float cosR = cos(angleCam.z);
	float sinR = sin(angleCam.z);

	// mat[col][row] by default
	rotMat[0][0] = + cosA * cosE;
	rotMat[1][0] = + cosA * sinE * sinR - sinA * cosR;
	rotMat[2][0] = + cosA * sinE * cosR + sinA * sinR;

	rotMat[0][1] = + sinA * cosE ;
	rotMat[1][1] = + sinA * sinE * sinR + cosA * cosR;
	rotMat[2][1] = + sinA * sinE * cosR - cosA * sinR;

	rotMat[0][2] = - sinE;
	rotMat[1][2] = + cosE * sinR;
	rotMat[2][2] = + cosE * cosR;

	return(rotMat);
}

void main() {
	// Camera rotation matrix
	mat3 rotMatrix = getRotMat(camAngle);

	vec2 equi_pts; // un-normalize latlong
	equi_pts[0] = (UV.x+.25) * (2*PI);
	equi_pts[1] = ((1 - UV.y) - .5) * PI;

	vec3 Pvf; // equi2sphere
	Pvf[0] = cos(equi_pts[1]) * sin(equi_pts[0]);
	Pvf[1] = cos(equi_pts[1]) * cos(equi_pts[0]);
	Pvf[2] = sin(equi_pts[1]);

	// sphere to U in world space (origin: camera rotation)
	vec3 Pvi = Pvf * rotMatrix;

	// Project on camera viewport
	Pvi[1] = Pvi[1] / (2 * tan(angleSpan.x/2.));
	Pvi[2] = Pvi[2] / (2 * tan(angleSpan.y/2.));

	// Unit world space to lat long
	vec2 VPpos;

	VPpos.x = atan(Pvi.y, Pvi.x);
	VPpos.y = PI-(asin(Pvi.z/length(Pvi)) + PI);

	VPpos = VPpos / angleSpan * 2;

	int inVP = int(abs(VPpos.x) < .5 && abs(VPpos.y) < .5);

	VPpos = VPpos+.5;
	colour = inVP * texture(VPTexture, VPpos.xy).xyz + (1-inVP) * texture(backgroundTexture, UV).rgb;
}
