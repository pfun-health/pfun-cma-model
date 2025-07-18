"""
[Blender and Python] Generating a procedural solar system with Blender's Python API
Mina Pêcheux - August 2021
Email: mina.pecheux@gmail.com

A very basic Blender script that creates a simplified solar system with a few
planets around a sun, adds shaders computed on-the-fly and sets various animation
curves on the objects to get random revolution speeds.

This code is a simple example of how to instantiate objects for
procedural generation in Blender using the Python API.

Read the full tutorial on Medium:
https://medium.com/geekculture/generating-a-procedural-solar-system-with-blenders-python-api-9754ece0cf03

--------

MIT License

Copyright (c) 2021 Mina Pêcheux

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from math import pi
from random import random
import bpy


def create_sphere(radius, distance_to_sun, obj_name):
    # instantiate a UV sphere with a given
    # radius, at a given distance from the
    # world origin point
    obj = bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=(distance_to_sun, 0, 0),
        scale=(1, 1, 1)
    )
    # rename the object
    bpy.context.object.name = obj_name
    # apply smooth shading
    bpy.ops.object.shade_smooth()
    # return the object reference
    return bpy.context.object


def create_torus(radius, obj_name):
    # (same as the create_sphere method)
    obj = bpy.ops.mesh.primitive_torus_add(
        location=(0, 0, 0),
        major_radius=radius,
        minor_radius=0.1,
        major_segments=60
    )
    bpy.context.object.name = obj_name
    # apply smooth shading
    bpy.ops.object.shade_smooth()
    return bpy.context.object


def create_emission_shader(color, strength, mat_name):
    # create a new material resource (with its
    # associated shader)
    mat = bpy.data.materials.new(mat_name)
    # enable the node-graph edition mode
    mat.use_nodes = True
    
    # clear all starter nodes
    nodes = mat.node_tree.nodes
    nodes.clear()

    # add the Emission node
    node_emission = nodes.new(type="ShaderNodeEmission")
    # (input[0] is the color)
    node_emission.inputs[0].default_value = color
    # (input[1] is the strength)
    node_emission.inputs[1].default_value = strength
    
    # add the Output node
    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    
    # link the two nodes
    links = mat.node_tree.links
    link = links.new(node_emission.outputs[0], node_output.inputs[0])

    # return the material reference
    return mat


def delete_object(name):
    # try to find the object by name
    if name in bpy.data.objects:
        # if it exists, select it and delete it
        obj = bpy.data.objects[name]
        obj.select_set(True)
        bpy.ops.object.delete(use_global=False)


def find_3dview_space():
    # Find 3D_View window and its scren space
    area = None
    for a in bpy.data.window_managers[0].windows[0].screen.areas:
        if a.type == "VIEW_3D":
            area = a
            break
    return area.spaces[0] if area else bpy.context.space_data


def setup_scene():
    # (set a black background)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    # (make sure we use the EEVEE render engine + enable bloom effect)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
#    import pdb
#    pdb.set_trace()

    # (set the animation start/end/current frames)
    scene.frame_start = START_FRAME
    scene.frame_end = END_FRAME
    scene.frame_current = START_FRAME
    # get the current 3D view (among all visible windows
    # in the workspace)
    space = find_3dview_space()
    # apply a "rendered" shading mode + hide all
    # additional markers, grids, cursors...
    space.shading.type = 'RENDERED'
    space.overlay.show_floor = False
    space.overlay.show_axis_x = False
    space.overlay.show_axis_y = False
    space.overlay.show_cursor = False
    space.overlay.show_object_origins = False


N_PLANETS = 7 
START_FRAME = 0
END_FRAME = 256

# setup scene settings
setup_scene()

# clean scene + planet materials
delete_object("Sun")
for n in range(N_PLANETS):
    delete_object("Planet-{:02d}".format(n))
    delete_object("Radius-{:02d}".format(n))
for m in bpy.data.materials:
    bpy.data.materials.remove(m)

ring_mat = create_emission_shader(
    (1, 1, 1, 1), 1, "RingMat"
)
for n in range(N_PLANETS):
    # get a random radius (a float in [1, 5])
    r = 1 + random() * 4
    # get a random distace to the origin point:
    # - an initial offset of 30 to get out of the sun's sphere
    # - a shift depending on the index of the planet
    # - a little "noise" with a random float
    d = 30 + n * 12 + (random() * 4 - 2)
    # instantiate the planet with these parameters
    # and a custom object name
    planet = create_sphere(r, d, "Planet-{:02d}".format(n))
    planet.data.materials.append(
        create_emission_shader(
            (random(), random(), 1, 1),
            2,
            "PlanetMat-{:02d}".format(n)
        )
    )
    # add the radius ring display
    ring = create_torus(d, "Radius-{:02d}".format(n))
    ring.data.materials.append(ring_mat)

    # set planet as active object
    bpy.context.view_layer.objects.active = planet
    planet.select_set(True)
    # set object origin at world origin
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")
    
    # setup the planet animation data
    planet.animation_data_create()
    planet.animation_data.action = bpy.data.actions.new(name="RotationAction")
    fcurve = planet.animation_data.action.fcurves.new(
        data_path="rotation_euler", index=2
    )
    k1 = fcurve.keyframe_points.insert(
        frame=START_FRAME,
        value=0
    )
    k1.interpolation = "LINEAR"
    rotation_value = (2 + random() * 2) * pi
    print('rotation_value=', rotation_value)
    k2 = fcurve.keyframe_points.insert(
        frame=END_FRAME,
        value=rotation_value
    )
    k2.interpolation = "LINEAR"

# add the sun sphere
sun = create_sphere(12, 0, "Sun")
sun.data.materials.append(
    create_emission_shader(
        (1, 0.66, 0.08, 1), 10, "SunMat"
    )
)

# deselect all objects
bpy.ops.object.select_all(action='DESELECT')

# Play the animation
bpy.ops.screen.animation_play()