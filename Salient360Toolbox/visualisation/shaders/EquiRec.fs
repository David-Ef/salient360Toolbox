#version 130
#define PI 3.1415926535897932384626433832795

in vec2 UV;
uniform float brightness;
uniform sampler2D backgroundTexture;
uniform bool showBorder;
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

	if (!showBorder){
		colour = texture(backgroundTexture, UV).rgb;
		colour *= brightness;
		return;
	}

	mat3 rotMatrix = getRotMat(camAngle);
	float max_dist = 1 * PI/180;

	vec2 equi_pts; // un-normalize latlong
	equi_pts[0] = (UV.x+.25) * (2*PI);
	equi_pts[1] = ((1 - UV.y) - .5) * PI;

	vec3 Pvf; // equi2sphere
	Pvf[0] = cos(equi_pts[1]) * sin(equi_pts[0]);
	Pvf[1] = cos(equi_pts[1]) * cos(equi_pts[0]);
	Pvf[2] = sin(equi_pts[1]);

	// sphere to U	 in world space (origin: camera rotation)
	vec3 Pvi = Pvf * rotMatrix;

	// Unit world space to lat long
	vec2 VPpos;
	VPpos.x = atan(Pvi.y, Pvi.x);
	VPpos.y = acos(Pvi.x / length(Pvi.xz)) * -sign(Pvi.z);

	bool dist, distO, distI;
	// Outer VP border
	distO = (abs(VPpos.x) - angleSpan.x/2) < max_dist
		&& (abs(VPpos.y) - angleSpan.y/2) < max_dist;
	// Inner VP border
	distI = (abs(VPpos.x) - angleSpan.x/2) < (max_dist/2)
		&& (abs(VPpos.y) - angleSpan.y/2) < (max_dist/2);

	// VP border
	dist = distO && !distI;

	// Gaze position in viewport ([-1, 1], [-1, 1])
	vec2 gazepos = vec2(.15, .5) * (angleSpan/2);
	VPpos = VPpos - gazepos;
	// adjust Y dim to VP (anglespan) ratio
	VPpos.y = VPpos.y * angleSpan.y / angleSpan.x;

	int inn = int(dist);

	// VP border
	colour = inn * vec3(1, 0, 0) + (1-inn) * texture(backgroundTexture, UV).rgb;

	colour *= brightness;
}