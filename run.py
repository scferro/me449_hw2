import numpy as np
import modern_robotics as mr
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def IKinBodyIterates(Blist, M, T, thetalist0, output_filename='output.csv', eomg=0.001, ev=0.0001):
    """
    Based on modern_robotics library function IKinBody()
    Computes inverse kinematics in the body frame for an open chain robot
        and prints report after each iteration of N-R solver

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    """
    # Initialize variables
    d = [1]
    i = 0
    maxiterations = 100000
    thetalist = np.array(thetalist0).copy()
    thetalist_list = [thetalist]
    linErrorList = []
    angErrorList = []
    pointsList = []
    Tsb_i = mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, thetalist)), T))
    Vb = mr.se3ToVec(Tsb_i)
    # Calculate initial error values
    angularErrorMag = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
    linearErrorMag = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
    angErrorList.append(angularErrorMag)
    linErrorList.append(linearErrorMag)
    err = angularErrorMag > eomg \
            or linearErrorMag > ev
    T_config = mr.FKinBody(M, Blist, thetalist)
    # Print an initial report
    print_report(i, not err, thetalist, T_config, Vb, angularErrorMag, linearErrorMag)
    while err and i < maxiterations:
        # Iterate the thetalist using the Newton-Raphson method
        thetalist = thetalist + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, thetalist)), Vb)
        i = i + 1
        thetalistmod = []
        for theta in thetalist:
            while theta > np.pi:
                theta += -2*np.pi
            while theta < -np.pi:
                theta += 2*np.pi
            thetalistmod.append(theta)
        thetalist = thetalistmod
        # Evaulate new EE configuration and error values
        Tsb_i = mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, thetalist)), T))
        Vb = mr.se3ToVec(Tsb_i)
        angularErrorMag = (Vb[0]**2 + Vb[1]**2 + Vb[2]**2)**0.5
        linearErrorMag = (Vb[3]**2 + Vb[4]**2 + Vb[5]**2)**0.5
        angErrorList.append(angularErrorMag)
        linErrorList.append(linearErrorMag)
        # Check if desired state is reached
        err = angularErrorMag > eomg \
              or linearErrorMag > ev
        # Add the current list of theta values to the array with all previous iterations of the list
        thetalist_list.append(thetalist)
        d.append(1)
        T_config = mr.FKinBody(M, Blist, thetalist)
        newPoint = [T_config[0][3],
                      T_config[1][3],
                      T_config[2][3]]
        pointsList.append(newPoint)
        # Print report
        print_report(i, not err, thetalist, T_config, Vb, angularErrorMag, linearErrorMag)
    # Write the list of thetalists to a CSV file
    write_to_csv(thetalist_list, output_filename)
    return (thetalist, angErrorList, linErrorList, not err, pointsList)

def write_to_csv(array, file_name):
    """
    Writes an array to a csv file

    :param array: The array to be written to the file
    :param M: The filename to be used
    """
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(array)

def print_report(iteration, solFound, jointVector, Tsb_i, error_twist, omega_b, v_b):
    """
    Prints a report of the current status of the N-R solver to the console

    :param iteration: The current iteration of the solver
    :param solFound: Whether the solution is found or not (bool)
    :param jointVector: The current joint vector 
    :param Tsb_i: The current configuration of the end effector
    :param error_twist: The error velcotiy vector
    :param omega_b: Magnitude of angular velocity error
    :param v_b: Magnitude of linear velocity error
    """
    print()
    print(f"Iteration {iteration}:")
    print()
    print(f"Solution Found: {solFound}")
    print()
    print('Joint Vector:')
    print(jointVector)
    print()
    print("SE(3) End-Effector Config:")
    print(Tsb_i)
    print(f"error_twist V_b: {error_twist}")
    print(f"Angular Error: {omega_b}")
    print(f"Linear Error: {v_b}")

# Define configuration variables and matrices
W1 = 0.109
W2 = 0.082
L1 = 0.425
L2 = 0.392
H1 = 0.089
H2 = 0.095

Blist = np.array([[0, 1, 0, W1+W2, 0, L1+L2],
                  [0, 0, 1, H2, -L1-L2, 0],
                  [0, 0,  1, H2, -L2, 0],
                  [0, 0, 1, H2, 0, 0],
                  [0, -1, 0, -W2, 0, 0],
                  [0, 0, 1, 0, 0, 0]]).T

M = np.array([[-1, 0,  0, L1+L2],
                [ 0, 0,  1, W1+W2],
                [ 0, 1, 0, H1-H2],
                [ 0, 0,  0, 1]])

T = np.array([[0.7071, 0,  0.7071, -0.3],
                [0.7071, 0, -0.7071, -0.5],
                [0, 1, 0, 0.5],
                [0, 0, 0, 1]])

thetalist0_long = np.array([1.324905074858755, -0.37122381441207536, 1.5925853354960249, 2.6764961135921777, 0.37695726754698367, 1.6362280196496837])
thetalist0_short = np.array([1.4555524590399056, -1.3484752037977774, -2.107773587920342, -2.169810786959951, -3.016705027011944, 0.6516665528815917])

# Run the functions
print()
print('Running the "long" solution.')
print()
thetalistLong, angErrorListLong, linErrorListLong, errLong, pointsListLong = IKinBodyIterates(Blist, M, T, thetalist0_long, 'long_iterates.csv')
print('Running the "short" solution.')
print()
thetalistShort, angErrorListShort, linErrorListShort, errShort, pointsListShort = IKinBodyIterates(Blist, M, T, thetalist0_short, 'short_iterates.csv')

# Break out data to plot
xListLong = np.array([])
yListLong = np.array([])
zListLong = np.array([])
xListShort = np.array([])
yListShort = np.array([])
zListShort = np.array([])
for point in pointsListLong:
    xListLong = np.append(xListLong, point[0])
    yListLong = np.append(yListLong, point[1])
    zListLong = np.append(zListLong, point[2])
for point in pointsListShort:
    xListShort = np.append(xListShort, point[0])
    yListShort = np.append(yListShort, point[1])
    zListShort = np.append(zListShort, point[2])

# Plot the data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(xListLong, yListLong, zListLong, label='Long Evaluation Result')
ax.plot(xListShort, yListShort, zListShort, label='Short Evaluation Result')
ax.scatter(xListLong[0], yListLong[0], zListLong[0], label='Long Evaluation Start', color='b')
ax.scatter(xListShort[0], yListShort[0], zListShort[0], label='Short Evaluation Start', color='r')
ax.scatter(xListShort[-1], yListShort[-1], zListShort[-1], label='Goal', color='g')

# Set labels
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Newton Raphson Solution Iterations for EE in 3D Space')

# Show plot
plt.legend()
plt.show()