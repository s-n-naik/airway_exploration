# testing
from turtle import *

# To save to a file, you can use postscript.

from tkinter import *
from turtle import *
import turtle
def demo_turtle_position():
    print(turtle.pos())

    # turtle move forward
    # by 40 pixels
    turtle.forward(40)

    # print position (after move)
    # i.e; (150.0, 0.0)
    print(turtle.position())

    # turtle move forward by 40 pixels
    # after taking right turn
    # by 45 degrees
    turtle.right(45)
    turtle.forward(40)

    # print position
    # (after next move)
    print(turtle.pos())

    # turtle move forward by 80
    # pixels after taking left
    # turn by 90 degrees
    turtle.left(90)
    turtle.forward(80)

    # print position
    # (after next move)
    print(turtle.pos())

    # turtle move forward
    # by 40 pixels after taking
    # right turn by 90 degrees
    turtle.right(90)
    turtle.forward(40)

    # print position (after next move)
    print(turtle.position())

    # turtle move forward by
    # 40 pixels after taking
    # left turn by 45 degrees
    turtle.left(45)
    turtle.forward(40)

    # print position
    # (after final move)
    print(turtle.pos())
    return None

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

D = 90
L = 10

def iterate(axiom, num=0, initator='F'):
    """
    Compute turtle rule string by iterating on an axiom
    """

    def translate(current, axiom):
        """
        Translate all the "F" with the axiom for current string
        """
        result = ''
        consts = {'+', '-', '[', ']'}
        for c in current:
            if c in consts:
                result += c
                continue
            if c == 'F':
                result += axiom
        return result

    # Set initator
    result = initator
    for i in range(0, num):
        # For ever iteration, translate the rule string
        result = translate(result, axiom)
    return result

def draw(axiom, d=D, l=L):
    """
    Use turtle to draw the L-System
    """
    stack  = []                 # For tracking turtle positions
    screen = turtle.Screen()
    alex   = turtle.Turtle()

    # alex.hideturtle()           # Don't show the turtle
    alex.speed(0)               # Make the turtle faster
    alex.left(90)               # Point up instead of right

    for i in range(len(axiom)):
        c = axiom[i]

        if c == 'F':
            alex.forward(l)


        if c == 'f':
            # alex.penup()
            alex.forward(l)
            # alex.pendown()

        if c == '+':
            alex.left(d)

        if c == '-':
            alex.right(d)

        if c == '[':
            stack.append((alex.heading(), alex.pos()))

        if c == ']':
            heading, position = stack.pop()
            # alex.penup()
            alex.goto(position)
            alex.setheading(heading)
            # alex.pendown()

    screen.onkey(screen.bye, 'q')
    screen.listen()
    turtle.mainloop()

if __name__ == '__main__':

    axiom = "FFFF"
    axiom = iterate(axiom, 100, "F[+F]F")
    draw(axiom, 90, 10)

#
# if __name__ == "__main__":
#     canvas = turtle_3d_cube()
#     print(canvas)