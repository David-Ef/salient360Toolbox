#version 130
#define PI 3.1415926535897932384626433832795

in vec2 UV;
uniform sampler2D backgroundTexture;
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
	rotMat[1][0] = + sinA * cosE;
	rotMat[2][0] = - sinE;

	rotMat[0][1] = + cosA * sinE * sinR - sinA * cosR;
	rotMat[1][1] = + sinA * sinE * sinR + cosA * cosR;
	rotMat[2][1] = + cosE * sinR;

	rotMat[0][2] = + cosA * sinE * cosR + sinA * sinR;
	rotMat[1][2] = + sinA * sinE * cosR - cosA * sinR;
	rotMat[2][2] = + cosE * cosR;

	return(rotMat);
}

void main() {
	mat3 rotMatrix = getRotMat(camAngle);

	float focal = 1.50;
	vec2 Dimpixels = vec2(1, 1);
	vec2 pxSize = vec2(
	(2 * focal * tan(angleSpan.x/2.)) / Dimpixels.x,
	(2 * focal * tan(angleSpan.y/2.)) / Dimpixels.y
	);
	vec2 ptsVP = UV * Dimpixels;

	vec3 Pvi = vec3(
		focal,
		pxSize.x * (ptsVP.x - Dimpixels.x / 2.),
		pxSize.y * (ptsVP.y - Dimpixels.y / 2.)
	);

	vec3 Pvf = (Pvi * rotMatrix);

	// Normalize to unit vector if you don't divide by length(Pvf)) in asin func.
	// Pvf = Pvf/length(Pvf);

	vec2 pts = vec2(0);
	pts[0] = atan(Pvf.x, Pvf.y) / (2*PI) -.25;
	pts[1] = 1-(asin(Pvf.z/length(Pvf)) / PI + .5);

	pts = pts + ivec2(int(pts[0]<0), int(pts[1]<0));

	vec2 border_dist = abs(vec2(1, 2)/2 - UV*vec2(1, 2));
	bool border = (border_dist.x > .49) || (border_dist.y > .975);

	colour = int(border) * vec3(1, 0, 0) + int(!border) * texture2D(backgroundTexture, pts.xy).rgb; // Display viewport
}