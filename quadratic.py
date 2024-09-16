import math 

def quadratic_func(a, b, c, x):
    a = float(input("Enter a: "))
    b = float(input("Enter b: "))
    c = float(input("Enter c: "))

    delta = b**2 - 4 * a * c

    if delta < 0:
        return "No real roots"
    elif delta == 0:
        return -b / (2 * a)
    else:
        return (-b + math.sqrt(delta)) / (2 * a), (-b - math.sqrt(delta)) / (2 * a)
    
    

    return a * x**2 + b * x + c