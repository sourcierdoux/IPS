from beacon import *
import numpy as np
from scipy.optimize import least_squares
import math

def d_points(x1,y1,x2,y2):
    #Returns the distance between two points
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def fun(pos, beacon_positions, distances):
    #The function to minimize for least square optimization
    x, y = pos
    return np.sqrt((beacon_positions[:, 0] - x)**2 +(beacon_positions[:, 1] - y)**2) - distances


def locate_square(b1: beacon, b2: beacon, b3: beacon, distances: np.array([])):
    #Function to test least square optimization for localization

    beacon_positions = np.array([
    [b1.x, b1.y],
    [b2.x, b2.y],
    [b3.x, b3.y]
])
    initial_guess = [4, 4]  # or any reasonable guess you might have
    try:
        result = least_squares(fun, initial_guess, args=(beacon_positions, distances), bounds=([0,0],[8,8]))
    except Exception as e:
        return 0,0
    user_x, user_y= result.x
    print(f"User's position using optimization problem: ({user_x}, {user_y})")
    return user_x, user_y

def locate_weight(b1: beacon,b2: beacon,b3: beacon):
    # Naive weight localization
    center1=np.array([b1.x,b1.y])
    center2=np.array([b2.x,b2.y])
    center3=np.array([b3.x,b3.y])
    inter1_2=get_intersections(b1.x,b1.y,b1.d_2D,b2.x,b2.y,b2.d_2D)
    if inter1_2==None:
        d = np.linalg.norm(center1 - center2)
        P1 = center1 + b1.d_2D * (center2 - b2.d_2D) / d
        P2 = center2 - b2.d_2D * (center2 - center1) / d
    
        # Calculate the midpoint between P1 and P2
        M = (P1 + P2) / 2
        I3=M
    else:
        x1,y1,x2,y2=inter1_2 #intersections between 1 and 2
        if d_points(x1,y1,b3.x,b3.y)<b3.d_2D:
            I3=(x1,y1)
        else:
            I3 = (x2,y2)

    
    inter1_3=get_intersections(b1.x,b1.y,b1.d_2D,b3.x,b3.y,b3.d_2D) #intersections between 1 and 3
    if inter1_3 == None:
    
        d = np.linalg.norm(center1 - center3)
        P1 = center1 + b1.d_2D * (center3 - b3.d_2D) / d
        P2 = center3 - b3.d_2D * (center3 - center1) / d
    
        # Calculate the midpoint between P1 and P2
        M = (P1 + P2) / 2
        I2=M
    else:
        x1,y1,x2,y2=inter1_3
        if d_points(x1,y1,b2.x,b2.y)<b2.d_2D:
            I2=(x1,y1)
        else:
            I2 = (x2,y2)
    inter2_3=get_intersections(b2.x,b2.y,b2.d_2D,b3.x,b3.y,b3.d_2D) #intersections between 2 and 3
    if inter2_3==None:
        d = np.linalg.norm(center2 - center3)
        P1 = center2 + b2.d_2D * (center3 - b3.d_2D) / d
        P2 = center3 - b3.d_2D * (center3 - center2) / d
    
        # Calculate the midpoint between P1 and P2
        M = (P1 + P2) / 2
        I1=M
    else:
        x1,y1,x2,y2=inter2_3
        if d_points(x1,y1,b1.x,b1.y)<b1.d_2D:
            I1=(x1,y1)
        else:
            I1 = (x2,y2)
        
    return (I1[0]+I2[0]+I3[0])/3, (I1[1]+I2[1]+I3[1])/3

def get_intersections(x0, y0, r0, x1, y1, r1, number_mode=False):
    # Returns either the number of intersections (number_mode = True) or the coordinates of intersections between two circles
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=d_points(x1,y1,x0,y0)
    # non intersecting
    if d > r0 + r1 :
        if number_mode==True:
            return 0      
        else:
            return None
    # One circle within other
    if d < abs(r0-r1):
        if number_mode==True:
            return 0
        else:
            return None
    # coincident circles
    if (d==(r0 + r1)):
        if number_mode==True:
            return 1
        else:
            a = (r0**2 - r1**2 + d**2) / (2 * d)
            x2 = x0 + a * (x1 - x0) / d
            y2 = y0 + a * (y1 - y0) / d
            return (x2, y2)
    else:
        if number_mode==True:
            return 2
        else:
            a=(r0**2-r1**2+d**2)/(2*d)
            h=math.sqrt(r0**2-a**2)
            x2=x0+a*(x1-x0)/d   
            y2=y0+a*(y1-y0)/d   
            x3=x2+h*(y1-y0)/d     
            y3=y2-h*(x1-x0)/d 

            x4=x2-h*(y1-y0)/d
            y4=y2+h*(x1-x0)/d
            return (x3, y3, x4, y4)

def sort_beacons_by_b_2D(b1: beacon, b2: beacon, b3: beacon):
    #Returns list of beacons ordered by their radiuses
    beacons = [b1, b2, b3]
    beacons.sort(key=lambda beacon: beacon.d_2D, reverse=False)
    return beacons

def is_circle_within(b1: beacon, b2: beacon):
    #True if circle of b1 is contained in b2, False otherwise
    return d_points(b1.x,b1.y,b2.x,b2.y)+b1.d_2D <= b2.d_2D

def any_contained(b1: beacon, b2: beacon, b3: beacon):
    #True if any beacon is contained within another, False otherwise
    if is_circle_within(b1,b2) or is_circle_within(b2,b1):
        return True
    if is_circle_within(b1,b3) or is_circle_within(b3,b1):
        return True
    if is_circle_within(b2,b3) or is_circle_within(b3,b2):
        return True
    return False

def select_case(b1: beacon, b2: beacon, b3: beacon):
    #Function to treat all cases of beacon radiuses


    n_intersections=0
    n_inter12=get_intersections(b1.x, b1.y, b1.d_2D, b2.x, b2.y, b2.d_2D,number_mode=True)
    n_inter13=get_intersections(b1.x,b1.y,b1.d_2D,b3.x,b3.y,b3.d_2D,number_mode=True)
    n_inter23=get_intersections(b2.x,b2.y,b2.d_2D,b3.x,b3.y,b3.d_2D, number_mode=True)
    n_intersections=n_intersections+n_inter12+n_inter23+n_inter13

    if b1.d_2D==0.2:
        return b1.x,b1.y
    if b2.d_2D==0.2:
        return b2.x, b2.y
    if b3.d_2D==0.2:
        return b3.x,b3.y

    if n_intersections==0 and any_contained(b1,b2,b3)==False:
        #Case with 0 intersections, and no circle is contained within another
        # Let C2 be the biggest circle, C3 the smallest and C1 the middle one
        C3, C1, C2 = sort_beacons_by_b_2D(b1, b2, b3)
        w31=C3.d_2D/C1.d_2D
        w32=C3.d_2D/C2.d_2D
        a=d_points(C3.x,C3.y,C1.x,C1.y)-C1.d_2D-C3.d_2D
        
        d1=a*w31
        
        P1x=C3.x+(C1.d_2D*(C1.x-C3.x))/d_points(C3.x,C3.y,C1.x,C1.y)
        P1y=C1.d_2D*(C1.y-C3.y)/d_points(C3.x,C3.y,C1.x,C1.y)+C3.y
        vx=(C1.x-P1x)*d1/(a+C1.d_2D)+P1x
        vy=(C1.y-P1y)*d1/(a+C1.d_2D)+P1y
        b=d_points(vx,vy,C2.x,C2.y)
        d2=b*w32
        Ax=d2*(C2.x-vx)/(b+C2.d_2D)+vx
        Ay=d2*(C2.y-vy)/(b+C2.d_2D)+vy
        return Ax, Ay
    if n_intersections==2 and any_contained(b1,b2,b3)==False:
        #Case where two circles intersect, but third one is far
        if n_inter12==2:
            C1,C3,C2=b1,b2,b3    
        elif n_inter13==2:
            C1,C2,C3=b1,b2,b3
        elif n_inter23==2:
            C1,C2,C3=b2,b1,b3
        else:
            raise Exception("error in the intersections")
        P1x, P1y, P2x, P2y = get_intersections(C1.x,C1.y,C1.d_2D,C3.x,C3.y,C3.d_2D,number_mode=False)
        d1=d_points(P1x,P1y,C2.x,C2.y)
        d2=d_points(P2x,P2y,C2.x,C2.y)
        if d2<d1:
            x=(C2.x-P2x)*(d2-C2.d_2D)/(2*d2)+P2x
            y=(C2.y-P2y)*(d2-C2.d_2D)/(2*d2)+P2y
        else:
            x=(C2.x-P1x)*(d1-C2.d_2D)/(2*d1)+P1x
            y=(C2.y-P1y)*(d1-C2.d_2D)/(2*d1)+P1y
        return x,y

    if n_intersections==4 and any_contained(b1,b2,b3)==False:
        #Case where one circle intersects with the two circles but they do not intersect 
        #We make sure C1 and C2 are the non-intersecting circles
        if n_inter12==0:
            C1,C2,C3=b1,b2,b3
        elif n_inter13==0:
            C1,C2,C3=b1,b3,b2
        elif n_inter23==0:
            C1,C2,C3=b2,b3,b1
        x1,y1,x2,y2=get_intersections(C1.x,C1.y,C1.d_2D,C3.x,C3.y,C3.d_2D)
        if d_points(x2,y2,C2.x,C2.y)<d_points(x1,y1,C2.x,C2.y):
            P1x,P1y=x2,y2
        else:
            P1x,P1y=x1,y1
        
        x3,y3,x4,y4=get_intersections(C3.x,C3.y,C3.d_2D,C2.x,C2.y,C2.d_2D)
        if d_points(x3,y3,C1.x,C1.y)<d_points(x4,y4,C1.x,C1.y):
            P2x,P2y=x3,y3       
        else:
            P2x,P2y=x4,y4
        return P1x+1/2*(P2x-P1x),P1y+1/2*(P2y-P1y)

    if n_intersections==6:
        #Case where all circles intersect in several points
        smallest_beacon_1, smallest_beacon_2, large_beacon= sort_beacons_by_b_2D(b1, b2, b3)
        x,y=another_case(smallest_beacon_2,large_beacon,smallest_beacon_1)
        return x,y
    
    
    if any_contained(b1,b2,b3):
        print("one circle is contained within another")
        return 0,0

def locate_specific_case(b1: beacon, b2: beacon, b3: beacon):
    #b3 should be the small circle
    print("\nRunning specific case")
    Ix,Iy,I_x,I_y=get_intersections(b1,b2)
    Jx,Jy,J_x,J_y=get_intersections(b1,b3)
    Kx,Ky,K_x,K_y=get_intersections(b2,b3)
    if d_points(J_x,J_y,K_x,K_y)<d_points(Jx,Jy,Kx,Ky):
        return (J_x+K_x)/2, (J_y+K_y)/2
    elif d_points(J_x,J_y,K_x,K_y)>d_points(Jx,Jy,Kx,Ky):
        return (Jx+Kx)/2, (Jy+Ky)/2

def locate_general_case(b1: beacon, b2: beacon, b3: beacon):
    print("Running general case")
    a_1=-2*b1.x
    b_1=-2*b1.y
    c_1=np.power(b1.x,2)+np.power(b1.y,2)-np.power(b1.d_2D,2)

    a_2=-2*b2.x
    b_2=-2*b2.y
    c_2=np.power(b2.x,2)+np.power(b2.y,2)-np.power(b2.d_2D,2)

    a_3=-2*b3.x
    b_3=-2*b3.y
    c_3=np.power(b3.x,2)+np.power(b3.y,2)-np.power(b3.d_2D,2)


    x = ((c_2-c_1)*(b_2-b_3)-(c_3-c_2)*(b_1-b_2))/((a_1-a_2)*(b_2-b_3)-(a_2-a_3)*(b_1-b_2))
    y = ((a_2-a_1)*(c_3-c_2)-(c_2-c_1)*(a_2-a_3))/((a_1-a_2)*(b_2-b_3)-(a_2-a_3)*(b_1-b_2))

    return x,y

def another_case(b1: beacon, b2: beacon, b3: beacon):
    #b1 and b3 should be the smallest circles
    x0,y0,x1,y1=get_intersections(b1.x,b1.y,b1.d_2D,b2.x,b2.y,b2.d_2D,number_mode=False)
    if d_points(x0,y0,b3.x,b3.y)<b3.d_2D:
        P1x, P1y = x0, y0
    else:
        P1x, P1y = x1, y1
    x0,y0,x1,y1=get_intersections(b1.x,b1.y,b1.d_2D,b3.x,b3.y,b3.d_2D,number_mode=False)
    if d_points(x0,y0,b2.x,b2.y)<b2.d_2D:
        P2x, P2y = x0, y0
    else:
        P2x, P2y = x1, y1
    x0,y0,x1,y1=get_intersections(b2.x,b2.y,b2.d_2D,b3.x,b3.y,b3.d_2D,number_mode=False)
    if d_points(x0,y0,b1.x,b1.y)<b1.d_2D:
        P3x, P3y = x0, y0
    else:
        P3x, P3y = x1, y1
    Px = (P1x+P2x+P3x)/3
    Py = (P1y + P2y + P3y)/3
    w32=b3.d_2D/b2.d_2D
    x=Px+(1-w32)*(P2x-Px)
    y=Py+(1-w32)*(P2y-Py)
    return x,y
