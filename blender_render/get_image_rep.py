#!/usr/bin/env blender --python
"""
Note how to properly pass an argument to blender:
https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
"""
import bpy
import os
import argparse
import sys


def set_active(obj):
    bpy.context.view_layer.objects.active = obj


def load_mesh(name):
    """Load mesh from our obj, and return the mesh name for future usage.

    Parameters
    ----------
    name: str
        Must be the full absolute path to the .obj file. We take its base name
        and strip the '.obj' at the end to get the mesh name that blender sees.
    """
    if name[-4:] == ".obj":
        bpy.ops.import_scene.obj(filepath=name)
        mesh_name = (os.path.basename(name)).replace('.obj','')
        return mesh_name
    else:
        raise ValueError("{} not an obj file".format(name))


def set_camera_pose():
    #TODO
    #Select the camera and make it the active object so that we can manipulate it
    camera = bpy.context.window.scene.objects['Camera']
    set_active(camera)

    #Set resolution
    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 480

    #Set the x, y and z location
    bpy.context.object.location[0] = 0.46195 #This value needs to be tweaked
    bpy.context.object.location[1] = -0.88443 #This value needs to be tweaked
    bpy.context.object.location[2] = 0.74862 #This value needs to be tweaked

    #Set the x, y and z rotation
    bpy.context.object.rotation_euler[0] = 1.0472 #This value needs to be tweaked
    bpy.context.object.rotation_euler[1] = -0.013421582 #This value needs to be tweaked
    bpy.context.object.rotation_euler[2] = -0.000595157275 #This value needs to be tweaked


def set_cloth_color(mesh_name):
    #TODO
    #Select the cloth and make it the active object
    cloth_obj = bpy.context.window.scene.objects[mesh_name]
    set_active(cloth_obj)

    #Don't use nodes (I don't know why but this makes it work)
    bpy.context.object.active_material.use_nodes = False

    #Set the base color
    bpy.context.object.active_material.diffuse_color = (0.8, 0.03, 0.05, 1) #Don't know what this color format is... need to figure it out

    #Need to do more in this function!
    return


def set_cloth_texture(mesh_name):
    #TODO
    return


def set_camera_focal_length():
    #TODO
    #Select the camera and make it the active object
    camera = bpy.context.window.scene.objects['Camera']
    set_active(camera)

    #Set the focal length
    bpy.context.object.data.lens = 40


def set_camera_optical_center():
    #TODO
    #Select the camera and make it the active object
    camera = bpy.context.window.scene.objects['Camera']
    set_active(camera)

    #Set the optical center (x,y) (I think this is the optical center but I'm not actually sure)
    bpy.context.object.data.shift_x = 0.0
    bpy.context.object.data.shift_y = 0.0


def set_lighting():
    #TODO
    pass


def render_image(obj_path):
    #TODO
    #Render the image
    img_path = obj_path.replace('.obj','.png')
    bpy.ops.render.render()
    bpy.data.images['Render Result'].save_render(filepath=img_path)


if __name__ == "__main__":
    # Get the .obj file as a command line argument, after a double dash.
    argv = sys.argv
    argv = argv[argv.index('--')+1:]
    assert len(argv) == 1, argv
    obj_path = argv[0]

    #TODO
    # Delete the starting cube
    bpy.ops.object.delete(use_global=False)

    # Load cloth mesh and get its 'mesh name' from the 'bpy scene'.
    mesh_name = load_mesh(obj_path)

    #Select the mesh and make it the active object so that we can manipulate it
    cloth_obj = bpy.context.window.scene.objects[mesh_name]
    set_active(cloth_obj)

    #Set cloth x rotation to 0 (this is so that the cloth mesh is in the starting pose we want)
    bpy.context.object.rotation_euler[0] = 0

    #Make the cloth mesh smooth
    bpy.ops.object.shade_smooth()

    #------Now let's set the scene------

    #Set the camera pose
    set_camera_pose()

    #Set the cloth_color
    set_cloth_color(mesh_name)

    #Set the cloth texture (will do this after I finish the other ones)
    #set_cloth_texture()

    #Set the camera focal length
    set_camera_focal_length()

    #Set the camera optical center
    set_camera_optical_center()

    #Set the lighting
    #set_lighting()

    #Render the image
    render_image(obj_path)
