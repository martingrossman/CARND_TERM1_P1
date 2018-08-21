import matplotlib.pyplot as plt
import numpy as np


def get_kqphi(point1: tuple, point2: tuple)->tuple:
    # from points point1, B get k, q, phi [rad]
    # y = k*x + q
    dy = point1[1] - point2[1]
    dx = point1[0] - point2[0]
    if abs(dx) < 0.00000001:
        k = float('Inf')
        q = None
        phi = np.pi/2
    elif abs(dy) < 0.00000001:
        k = 0
        q = point1[1] - k * point1[0]
        phi = 0
    else:
        k = dy/dx
        q = point1[1] - k * point1[0]
        phi = np.arctan2(dy, dx)

    return k, q, phi

xs = 40
ys = 30

A = [10, 20]
B = [15, 5]

k, q, phi = get_kqphi(A, B)
xb = (ys-q)/k
xt = -q/k

print(xb)

print('k={}, q={}, phi={}'.format(k, q, phi))


def plot_line(p1, p2, xsize=40.0, ysize=30.0):

    plt.figure()
    plt.plot([p1[0], p2[0]],[p1[1], p2[1]])

    plt.plot([xb, xt],[ys,0],'ro')
    plt.xlim(0, xsize)
    plt.ylim(ysize, 0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_line(A, B, xs, ys)

#xb = (ys - )
# line = [A,B]
# points = [A,B]
# px =[p[0] for p in points]
# py =[p[1] for p in points]
# C = [2, 0]
# D = [10,0]
#
# # A = [0.5, 0]
# # B = [10.8, 9]
# # C = [2.5, 3]
# # D = [8,7]
#
# A = [0, 0]
# B = [1, 8]
# C = [0, 0]
# D = [8, 1]





#
#
# points = [A, B, C, D]
#
# px =[p[0] for p in points]
# py =[p[1] for p in points]
#
#
#
# line_0 = [A, B]
# line_1 = [C, D]
# #line_ok = [E, F]
# lines_in = [line_0, line_1]
# lines_in_kqphi = [get_kqphi(line[0], line[1]) for line in lines_in]
#
#
#
#
#
#
# # simple 1st degree fit
# line_result_polyfit_kq = np.polyfit(x=px, y=py, deg=1)
# #print(line_result_polyfit_kq)
#
#
# # total fit
# def f(B,x):
#     return B[0]*x + B[1]
#
# linear = odr.Model(f)
#
# mydata = odr.Data(x=px,y=py)
# myodr = odr.ODR(mydata, linear, beta0=[1., 2.])
# myoutput = myodr.run()
#
# line_result_totalfit_kq = myoutput.beta
#
#
# # k average fit
# lines_in_k_avg = np.sum([line[0] for line in lines_in_kqphi]) / len([line[0] for line in lines_in_kqphi])
# lines_in_q_avg = np.sum([line[1] for line in lines_in_kqphi]) / len([line[1] for line in lines_in_kqphi])
# line_result_kqAverage_kq = [lines_in_k_avg, lines_in_q_avg]
#
# # phi average fit
# lines_in_phi_avg = np.sum([line[2] for line in lines_in_kqphi]) / len([line[2] for line in lines_in_kqphi])
# k_avgphi = np.tan(lines_in_phi_avg)
# line_result_phiqAverage_kq = [k_avgphi, lines_in_q_avg]
#
#
# line_polyfit = [[0, line_result_polyfit_kq[0] * 0 + line_result_polyfit_kq[1]], [6, line_result_polyfit_kq[0] * 6 + line_result_polyfit_kq[1]]]
# line_totalfit = [[0, line_result_totalfit_kq[0] * 0 + line_result_totalfit_kq[1]], [6, line_result_totalfit_kq[0] * 6 + line_result_totalfit_kq[1]]]
# line_phifit = [[0, line_result_phiqAverage_kq[0] * 0 + line_result_phiqAverage_kq[1]], [6, line_result_phiqAverage_kq[0] * 6 + line_result_phiqAverage_kq[1]]]
# line_kfit = [[0, line_result_kqAverage_kq[0] * 0 + line_result_kqAverage_kq[1]], [4, line_result_kqAverage_kq[0] * 4 + line_result_kqAverage_kq[1]]]
#
# plt.figure()
# plt.plot(px, py,'ko')
# plt.plot([p[0] for p in line_0],[p[1] for p in line_0],'r')
# plt.plot([p[0] for p in line_1],[p[1] for p in line_1],'b')
# #plt.plot([p[0] for p in line_ok],[p[1] for p in line_ok],'g')
# plt.plot([p[0] for p in line_polyfit],[p[1] for p in line_polyfit],'k')
# plt.plot([p[0] for p in line_totalfit],[p[1] for p in line_totalfit],'cyan')
# plt.plot([p[0] for p in line_phifit],[p[1] for p in line_phifit],'green')
# plt.plot([p[0] for p in line_kfit],[p[1] for p in line_kfit],'magenta')
# plt.axis('equal')
# red_patch = mpatches.Patch(color='red', label='line from points  A,B')
# blue_patch = mpatches.Patch(color='blue', label='line from points  C,D')
# magenta_patch = mpatches.Patch(color='magenta', label='line frm avg of k')
# green_patch = mpatches.Patch(color='green', label='line frm avg of phi')
# cyan_patch = mpatches.Patch(color='cyan', label='line frm totalfit of ABCD pts')
# black_patch = mpatches.Patch(color='black', label='line frm polyfit of ABCD pts')
# plt.legend(handles=[red_patch, blue_patch,magenta_patch,green_patch,cyan_patch,black_patch])
# plt.title('Comparison of \"averaging\" of 2 lines given by 4 points')
# plt.show()

