import numpy as np


def is_inside_quad(point, quad):
    AB = (quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])
    BC = (quad[2][0] - quad[1][0], quad[2][1] - quad[1][1])
    CD = (quad[3][0] - quad[2][0], quad[3][1] - quad[2][1])
    DA = (quad[0][0] - quad[3][0], quad[0][1] - quad[3][1])
    AP = (point[0] - quad[0][0], point[1] - quad[0][1])
    BP = (point[0] - quad[1][0], point[1] - quad[1][1])
    CP = (point[0] - quad[2][0], point[1] - quad[2][1])
    DP = (point[0] - quad[3][0], point[1] - quad[3][1])
    AB_AP = AB[0] * AP[1] - AB[1] * AP[0]
    BC_BP = BC[0] * BP[1] - BC[1] * BP[0]
    CD_CP = CD[0] * CP[1] - CD[1] * CP[0]
    DA_DP = DA[0] * DP[1] - DA[1] * DP[0]
    if (AB_AP >= 0 and BC_BP >= 0 and CD_CP >= 0 and DA_DP >= 0) or (
            AB_AP <= 0 and BC_BP <= 0 and CD_CP <= 0 and DA_DP <= 0):
        return True
    return False


def numero_element(point, element_dict, nodes_dict):
    for i in element_dict:
        n1 = element_dict[i]['nodes'][0]
        n2 = element_dict[i]['nodes'][1]
        n3 = element_dict[i]['nodes'][2]
        n4 = element_dict[i]['nodes'][3]
        quad = [
            (nodes_dict[n1][0], nodes_dict[n1][1]),
            (nodes_dict[n2][0], nodes_dict[n2][1]),
            (nodes_dict[n3][0], nodes_dict[n3][1]),
            (nodes_dict[n4][0], nodes_dict[n4][1]),
        ]
        if is_inside_quad(point, quad):
            return True, i
    return False, 0


def interpolation_Hencky(point, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
    x = point[0]
    y = point[1]
    X = np.array([
        [4, x1+x2+x3+x4, y1+y2+y3+y4, x1*y1+x2*y2+x3*y3+x4*y4],
        [x1+x2+x3+x4, x1**2+x2**2+x3**2+x4**2, x1*y1+x2*y2+x3*y3+x4*y4,
         x1**2*y1+x2**2*y2+x3**2*y3+x4**2*y4],
        [y1+y2+y3+y4, x1*y1+x2*y2+x3*y3+x4*y4,
         y1**2+y2**2+y3**2+y4**2, x1*y1**2+x2*y2**2+x3*y3**2+x4*y4**2],
        [x1*y1+x2*y2+x3*y3+x4*y4,
         x1**2*y1+x2**2*y2+x3**2*y3+x4**2*y4,
         x1*y1**2+x2*y2**2+x3*y3**2+x4*y4**2,
         x1**2*y1**2+x2**2*y2**2+x3**2*y3**2+x4**2*y4**2],
    ])
    try:
        X_inv = np.linalg.inv(X)
    except np.linalg.LinAlgError:
        X_inv = np.linalg.pinv(X)
    coeffs = X_inv.dot(np.array([
        z1+z2+z3+z4,
        x1*z1+x2*z2+x3*z3+x4*z4,
        y1*z1+y2*z2+y3*z3+y4*z4,
        x1*y1*z1+x2*y2*z2+x3*y3*z3+x4*y4*z4,
    ]))
    return coeffs[0] + coeffs[1]*x + coeffs[2]*y + coeffs[3]*x*y
