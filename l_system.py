# # testing
from turtle import *
# import turtle
# from openalea import lpy
# l = lpy.Lsystem('lpy/models/ArchiModels/massart.lpy')
# # To save to a file, you can use postscript.
#
# from tkinter import *
# import openalea.lpy as lpy

import importlib.util
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
scene = lsystem.sceneInterpretation(lstring) #list(lsystem)[-1]
print("SCENE", scene)
scene.save("test_scene.jpg")
# # for lstring in lsystem:
#
# t = PglTurtle()
# lsystem.turtle_interpretation(lstring, t)
# # scene = t.Canvas
# # print(scene)
# scene = t.getScene()


def sc2dict(s):
    d = {}
    for i in s:
        if i.id not in d:
            d[i.id] = []
        d[i.id].append(i)
    return d


lcode = """
Axiom:A
production:
A --> A[+A]A[-A]
homomorphism:
maximum depth:2
A --> BCB
B --> F
C --> f@Of
endlsystem
"""
#
#
# def test_generateScene():
#     """ Test Lsystem generation of a scene using homomorphism """
#     l = Lsystem()
#     l.set(lcode)
#     a = l.iterate()
#     sc = l.sceneInterpretation(a)
#     assert len(sc) == 4*3
#     d = sc2dict(sc)
#     assert len(d) == 4
#     print(d)
#     return d
# if __name__ == '__main__':
#     d = test_generateScene()
#     print(d[0][0])
# # scene = generateScene(lstring, t)
# # Viewer.display(scene)
# # scene.save('test_scene.ply')
#
# # # print(t.Canvas)
# # Viewer.display(scene)
# # time.sleep(2)
#
# class Plotter:
#     def __init__(self):
#         self.selectionAsked = False
#     def display(self,sc):
#         pass
#     def selection(self):
#         if not self.selectionAsked:
#             print('selection')
#             self.selectionAsked = True
#             return [3]

    # print(scene.Canvas)

# from turtle import *
# import turtle
# def demo_turtle_position():
#     print(turtle.pos())
#
#     # turtle move forward
#     # by 40 pixels
#     turtle.forward(40)
#
#     # print position (after move)
#     # i.e; (150.0, 0.0)
#     print(turtle.position())
#
#     # turtle move forward by 40 pixels
#     # after taking right turn
#     # by 45 degrees
#     turtle.right(45)
#     turtle.forward(40)
#
#     # print position
#     # (after next move)
#     print(turtle.pos())
#
#     # turtle move forward by 80
#     # pixels after taking left
#     # turn by 90 degrees
#     turtle.left(90)
#     turtle.forward(80)
#
#     # print position
#     # (after next move)
#     print(turtle.pos())
#
#     # turtle move forward
#     # by 40 pixels after taking
#     # right turn by 90 degrees
#     turtle.right(90)
#     turtle.forward(40)
#
#     # print position (after next move)
#     print(turtle.position())
#
#     # turtle move forward by
#     # 40 pixels after taking
#     # left turn by 45 degrees
#     turtle.left(45)
#     turtle.forward(40)
#
#     # print position
#     # (after final move)
#     print(turtle.pos())
#     return None

def turtle_3d_cube():
    tur = turtle.Screen()

    tur.bgcolor("black")

    tur.title("Python Guides")
    turt = turtle.Turtle()

    turt.color("blue")
    tut = turtle.Screen()

    for i in range(4):
        turt.forward(100)
        turt.left(90)


    turt.goto(50, 50)
    print("here", turtle.position())

    for i in range(4):
        print(turtle.pos())
        turt.forward(100)
        turt.left(90)

    turt.goto(150, 50)
    print(turtle.pos())
    turt.goto(100, 0)

    turt.goto(100, 100)
    print(turtle.pos())
    turt.goto(150, 150)

    turt.goto(50, 150)
    turt.goto(0, 100)
    turtle.done()
    return turtle.Canvas
    # ts = turtle.getscreen()
#
# ts.getcanvas().postscript(file="polygon.eps")

# D = 90
# L = 10
#
# def iterate(axiom, num=0, initator='F'):
#     """
#     Compute turtle rule string by iterating on an axiom
#     """
#
#     def translate(current, axiom):
#         """
#         Translate all the "F" with the axiom for current string
#         """
#         result = ''
#         consts = {'+', '-', '[', ']'}
#         for c in current:
#             if c in consts:
#                 result += c
#                 continue
#             if c == 'F':
#                 result += axiom
#         return result
#
#     # Set initator
#     result = initator
#     for i in range(0, num):
#         # For ever iteration, translate the rule string
#         result = translate(result, axiom)
#     return result
#
# def draw(axiom, d=D, l=L):
#     """
#     Use turtle to draw the L-System
#     """
#     stack  = []                 # For tracking turtle positions
#     screen = turtle.Screen()
#     alex   = turtle.Turtle()
#
#     # alex.hideturtle()           # Don't show the turtle
#     alex.speed(0)               # Make the turtle faster
#     alex.left(90)               # Point up instead of right
#
#     for i in range(len(axiom)):
#         c = axiom[i]
#
#         if c == 'F':
#             alex.forward(l)
#
#
#         if c == 'f':
#             # alex.penup()
#             alex.forward(l)
#             # alex.pendown()
#
#         if c == '+':
#             alex.left(d)
#
#         if c == '-':
#             alex.right(d)
#
#         if c == '[':
#             stack.append((alex.heading(), alex.pos()))
#
#         if c == ']':
#             heading, position = stack.pop()
#             # alex.penup()
#             alex.goto(position)
#             alex.setheading(heading)
#             # alex.pendown()
#
#     screen.onkey(screen.bye, 'q')
#     screen.listen()
#     turtle.mainloop()
#
# if __name__ == '__main__':
#
#     axiom = "FFFF"
#     axiom = iterate(axiom, 100, "F[+F]F")
#     draw(axiom, 90, 10)
#
# #
# # if __name__ == "__main__":
# #     canvas = turtle_3d_cube()
# #     print(canvas)