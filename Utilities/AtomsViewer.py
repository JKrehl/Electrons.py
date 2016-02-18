# -*- coding: utf-8 -*-
# vispy: gallery 30
# -----------------------------------------------------------------------------
# 2014, Aurore Deschildre, Gael Goret, Cyrille Rossant, Nicolas P. Rougier.
# Distributed under the terms of the new BSD License.
# -----------------------------------------------------------------------------
# 2016 Jonas Krehl numerous changes
#------------------------------------------------------------------------------
import numpy

from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate, ortho, scale
from vispy.io import load_data_file

vertex = """
#version 120
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_light_position;
uniform vec3 u_light_spec_position;
uniform float u_zoom;
attribute vec3  a_position;
attribute vec3  a_color;
attribute float a_radius;
varying vec3  v_color;
varying vec4  v_eye_position;
varying float v_radius;
varying vec3  v_light_direction;
void main (void) {
	v_radius = a_radius;
	v_color = a_color;
	v_eye_position = u_view * u_model * vec4(a_position,1.0);
	v_light_direction = normalize(u_light_position);
	float dist = length(v_eye_position.xyz);
	gl_Position = u_projection * v_eye_position;
	// stackoverflow.com/questions/8608844/...
	//  ... resizing-point-sprites-based-on-distance-from-the-camera
	vec4  proj_corner = u_projection * vec4(a_radius, a_radius, v_eye_position.z, v_eye_position.w);  // # noqa
	gl_PointSize = u_zoom* proj_corner.x / proj_corner.w;
}
"""

fragment = """
#version 120
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_light_position;
uniform vec3 u_light_spec_position;
varying vec3  v_color;
varying vec4  v_eye_position;
varying float v_radius;
varying vec3  v_light_direction;
void main()
{
	// r^2 = (x - x0)^2 + (y - y0)^2 + (z - z0)^2
	vec2 texcoord = gl_PointCoord* 2.0 - vec2(1.0);
	float x = texcoord.x;
	float y = texcoord.y;
	float d = 1.0 - x*x - y*y;
	if (d <= 0.0)
		discard;
	float z = sqrt(d);
	vec4 pos = v_eye_position;
	pos.z += v_radius*z;
	vec3 pos2 = pos.xyz;
	pos = u_projection * pos;
//    gl_FragDepth = 0.5*(pos.z / pos.w)+0.5;
	vec3 normal = vec3(x,y,z);
	float diffuse = clamp(dot(normal, v_light_direction), 0.0, 1.0);
	// Specular lighting.
	vec3 M = pos2.xyz;
	vec3 O = v_eye_position.xyz;
	vec3 L = u_light_spec_position;
	vec3 K = normalize(normalize(L - M) + normalize(O - M));
	// WARNING: abs() is necessary, otherwise weird bugs may appear with some
	// GPU drivers...
	float specular = clamp(pow(abs(dot(normal, K)), 40.), 0.0, 1.0);
	vec3 v_light = vec3(1., 1., 1.);
	gl_FragColor.rgb = (.15*v_color + .55*diffuse * v_color
						+ .35*specular * v_light);
}
"""


class AtomsViewer(app.Canvas):

	def __init__(self, atoms, radius=6.):
		app.Canvas.__init__(self, title='Molecular viewer', keys='interactive', size=(1200, 800))
		self.ps = self.pixel_scale

		self.zoom = .5/numpy.amax(numpy.linalg.norm(atoms['zyx'], axis=1))

		self.program = gloo.Program(vertex, fragment)
		self.view = scale(self.zoom*numpy.ones(3))
		self.model = numpy.eye(4, dtype=numpy.float32)
		self.projection = numpy.eye(4, dtype=numpy.float32)

		data = numpy.zeros(atoms.size, [('a_position', numpy.float32, 3),
							('a_color', numpy.float32, 4),
							('a_radius', numpy.float32, 1)])

		data['a_position'] = atoms['zyx']
		data['a_color'] = 1,0,0,1
		data['a_color'][atoms['Z']==16] = 1,1,0,1
		data['a_color'][atoms['Z']==74] = 0,.5,1,1
		data['a_radius'] = atoms['Z']**(1/3)*radius/self.zoom

		self.program.bind(gloo.VertexBuffer(data))

		self.program['u_zoom'] = self.zoom
		self.program['u_model'] = self.model
		self.program['u_view'] = self.view
		self.program['u_light_position'] = 0., 0., 2.
		self.program['u_light_spec_position'] = -5., 5., -5.
		self.apply_zoom()

		self.program['u_model'] = self.model
		self.program['u_view'] = self.view

		gloo.set_state(depth_test=True, clear_color='white')
		self.show()

	def run(self):
		app.run()

	def on_resize(self, event):
		self.apply_zoom()

	def apply_zoom(self):
		width, height = self.physical_size
		gloo.set_viewport(0, 0, width, height)
		a, b, c = .5, .5, 2
		if width>height:
			a *= width/height
		else:
			b *= height/width
		self.projection = ortho(-a, a, -b, b, -c, c)
		self.program['u_projection'] = self.projection

	def on_mouse_move(self, event):
		if event.is_dragging and event.button==1:
			delta = .2*(event.pos - event.last_event.pos)
			self.model = numpy.dot(self.model, rotate(delta[0], (0, 1, 0)))
			self.model = numpy.dot(self.model, rotate(delta[1], (1, 0, 0)))

			self.program['u_model'] = self.model
			self.update()


	def on_key_press(self, event):
		pass

	def on_mouse_wheel(self, event):
		self.zoom *= .9**event.delta[1]
		self.view = scale(self.zoom*numpy.ones(3))

		self.program['u_zoom'] = self.zoom

		self.program['u_view'] = self.view
		self.update()

	def on_draw(self, event):
		gloo.clear()
		self.program.draw('points')