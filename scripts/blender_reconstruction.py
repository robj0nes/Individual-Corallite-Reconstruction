import pickle
import bpy
import bmesh
import math
import numpy as np
from datetime import datetime

# TODO: Parameterize the quality of the model (ie. vertices per ellipse)
# TODO: Figure out how to adjust model display/output to handle timelapse changes once constructed.

# Utility function to print to Blender console for debugging.
def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")


def add_ellipse(region, verts):
    bpy.ops.mesh.primitive_circle_add(vertices=verts, radius=1, location=region['loc'])
    #    print(f"adding ellipse at: {region['loc']}")
    bpy.ops.transform.resize(value=region['shape'])
    bpy.ops.transform.rotate(value=-region['orient'], orient_axis='Z')

    # To add faces to ellipses
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.edge_face_add()
    bpy.ops.object.editmode_toggle()

    z = region['loc'][2]
    bpy.context.active_object.name = f"corallite_id:{region['id']}_height:{z}"
    bpy.data.collections['InProgress'].objects.link(bpy.context.active_object)


def get_next_connections(bm_faces, start_height, mat, threshold):
    next_height = np.infty
    curr = []

    for face in bm_faces:
        if face[1] == start_height:
            curr.append(face)
        elif next_height > face[1] > start_height:
            next_height = face[1]
    next = [face for face in bm_faces if face[1] == next_height]

    conns = dict()
    for c in curr:
        conns[c[0]] = []
        cent = mat @ c[0].calc_center_bounds()
        for n in next:
            cent2 = mat @ n[0].calc_center_bounds()
            dist = math.sqrt((cent[0] - cent2[0]) ** 2 + (cent[1] - cent2[1]) ** 2)
            if dist < threshold:
                conns[c[0]].append(n)

    return curr, conns, next_height


def connect_corallite(id):
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Join all objects
    for o in bpy.data.collections['InProgress'].all_objects:
        #        print(f"Selecting {o}")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
    bpy.ops.object.join()

    # Enter edit mode
    bpy.ops.object.editmode_toggle()

    # Get a BMesh representation of the active mesh
    obj = bpy.context.edit_object
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    mat = obj.matrix_world

    # Build a list of faces, sorted by height
    bm_faces = []
    for face in bm.faces:
        cent = mat @ face.calc_center_bounds()
        #        print(f"adding cent: {cent}")
        bm_faces.append([face, round(cent.z, 2)])
    bm_faces.sort(key=lambda x: x[1])

    #    for i in bm_faces:
    #        print(i)

    # Iterate through the layers connecting faces with their NN next layer.
    start_height = bm_faces[0][1]
    #    for i in range(23):
    for _ in range(len(bm_faces) - 1):
        bpy.ops.mesh.select_all(action='DESELECT')

        curr, conns, start_height = get_next_connections(bm_faces, start_height, mat, 0.15)
        wait = False
        for curr_face in curr:
            curr_face[0].select = True
            for next_conn in conns[curr_face[0]]:
                # Check for multiple connections to same next_conn
                # If so, select and wait until entire loop is complete
                for curr_face2 in curr:
                    if curr_face != curr_face2:
                        for next_conn2 in conns[curr_face2[0]]:
                            if next_conn == next_conn2:
                                curr_face2[0].select = True
                                wait = True
                next_conn[0].select = True
            # If only one curr_face connecting to the next_conn safe to connect
            if not wait:
                bpy.ops.mesh.bridge_edge_loops()
                bpy.ops.mesh.select_all(action='DESELECT')
        # If wait was flagged, its now safe to connect
        if wait:
            bpy.ops.mesh.bridge_edge_loops()
            bpy.ops.mesh.select_all(action='DESELECT')

    bpy.ops.object.editmode_toggle()

    # Move corallite to the 'done' collection
    bpy.data.collections['Made'].objects.link(bpy.context.active_object)
    bpy.data.collections['InProgress'].objects.unlink(bpy.context.active_object)


# Currently out of use, but saving in case required for 'time-line' approach
def build_by_layer(coral):
    for layer in coral["detections"]:
        for c in enumerate(layer):
            add_ellipse(c)
    for i in range(coral["total"]):
        connect_corallite(i)


def build(coral, verts):
#    for id in list(coral.keys()):
    for id in range(800, len(coral)):
        if id in list(coral.keys()):
            if coral[id] != []:
                for region in coral[id]:
#                for region in coral[id]['regions']:
                    add_ellipse(region, verts)
                connect_corallite(id)


def reset_scene():
    for col in bpy.data.collections:
        for obj in col.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    for col in bpy.data.collections:
        bpy.data.collections.remove(col)

    collection = bpy.data.collections.new("Made")
    bpy.context.scene.collection.children.link(collection)
    collection = bpy.data.collections.new("InProgress")
    bpy.context.scene.collection.children.link(collection)


if __name__ == '__main__':
    reset_scene()

    data_root = "/Users/rob/University/Publications/3D Individual Corallite " \
                "Reconstruction/Individual_Corallite_Reconstruction/data/example_species "
    file_name = f"{data_root}/blender_data/0.15nn_search_depth3_SMLdata.pkl"
    
    with open(file_name, 'rb') as f:
        coral = pickle.load(f)
    verts = 16
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    build(coral, verts)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
