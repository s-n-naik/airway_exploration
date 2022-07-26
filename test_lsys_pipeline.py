# # testing
from turtle import *
import turtle as turtle
# from openalea import lpy
# l = lpy.Lsystem('lpy/models/ArchiModels/massart.lpy')
# # To save to a file, you can use postscript.
import matplotlib.pyplot as plt
import os
# from tkinter import *
# import openalea.lpy as lpy
from openalea.plantgl.all import *
from imp import reload
# import openalea.plantgl.algo.view as view; reload(view)
from openalea.plantgl.algo.view import *
from openalea.plantgl.algo.view import *
import importlib.util

def view_2(scene: Scene, imgsize : tuple = (800,800), perspective : bool = False, zoom : float = 1, azimuth : float = 0 , elevation : float = 0, savepath=None, ax=None) -> None:
    """
    Display an orthographic view of a scene.
    :param scene: The scene to render
    :param imgsize: The size of the image
    :param azimuth: The azimuth (in degrees) of view to render
    :param elevation: The elevation (in degrees) of view to render
    """
    if perspective:
        img = perspectiveimage(scene, imgsize=imgsize, zoom=zoom, azimuth=azimuth, elevation=elevation)
    else:
        img = orthoimage(scene, imgsize=imgsize, azimuth=azimuth, elevation=elevation)
    if not img is None:
        # import matplotlib.pyplot as plt
        if ax is not None:
            ax.imshow(img, cmap='binary')
        else:
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.imshow(img)
            if savepath is not None:
                plt.savefig(os.path.abspath(savepath))
            plt.show()


def main():
    package= 'openalea.lpy'
    is_present = importlib.util.find_spec(package) #find_spec will look for the package
    if is_present is None:
        print(package +" is not installed")
    else:
        print ("Successfull")
main()
#
import time
from openalea.lpy import *
from openalea.plantgl.all import *

lsystem = Lsystem("basic_tree_v3.lpy")
# # lsystem.animate()
lstring = lsystem.derive()
print('string',lstring)
# print(lsystem.interpret(lstring))

# THIS WORKS BELOW!!! view(scene) - its static but works!
scene = lsystem.sceneInterpretation(lstring) #list(lsystem)[-1]
print("SCENE", scene)
# scene.save("test_scene.mtg")
azimuth = list(range(0,360, 90))
elevation = [0,-90]
print(list(azimuth))
f, axes = plt.subplots(len(elevation),len(list(azimuth)))
for i in range(len(list(azimuth))):
    for j in range(len(elevation)):
        # if i < len(azimuth)//2:
        #     x = 0
        #     y = i
        # else:
        #     x=1
        #     y=i - len(azimuth)//2
        az = azimuth[i]
        el = elevation[j]
        # ax = axes[x,y]
        ax = axes[j,i]
        ax.set_title(f'az: {az}, el:{elevation}')
        view_2(scene, perspective=False, azimuth=az,elevation=el, savepath=f'test_pic_az{az}_el{el}.png', ax=ax)

plt.savefig('all_pics_test.png')
plt.show()
