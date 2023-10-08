from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete import (FieldVariable, Material, Integral, Function, Equation, Equations, Problem)
from sfepy.mechanics.matcoefs import stiffness_from_lame, stiffness_from_youngpoisson
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.base.base import IndexedStruct
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton

import numpy as np

def calculate_displacement(filename, force_vertices_tags, constraints_vertices_tags, force):
    mesh = Mesh.from_file(filename)
    domain = FEDomain('domain', mesh)
    
    force_vertices_string = "vertex "
    for index in range(len(force_vertices_tags)):
        force_vertices_string += "{}".format(force_vertices_tags[index] - 1)
        if index != len(force_vertices_tags) - 1:
            force_vertices_string += ", "

    forces = domain.create_region('Forces', force_vertices_string, 'vertex')

    # constraints_vertices_string = ""

    # if(len(constraints_vertices_tags) == 0):
    #     pass
    # elif(len(constraints_vertices_tags) == 1):
    #     constraints_vertices_string = "vertex {}, {}".format(constraints_vertices_tags[0][0], constraints_vertices_tags[0][1])
    #     print(constraints_vertices_string)
    #     constraints = domain.create_region('Constraints', constraints_vertices_string, 'vertex')
    # else:
    #     constraints_vertices_string += "vertex "
    #     for index in range(len(constraints_vertices_tags)):
    #         constraints_vertices_string += "{}, {}".format(constraints_vertices_tags[index][0], constraints_vertices_tags[index][1])
    #         if index != len(constraints_vertices_tags) - 1:
    #             constraints_vertices_string += ","
    #     print(constraints_vertices_string)
    #     constraints = domain.create_region('Constraints', constraints_vertices_string, 'vertex')
    
    constraints = []

    def get_edge(coords, tags, domain = None):
        coord0 = coords[tags[0] - 1]
        coord1 = coords[tags[1] - 1]
        x1, y1 = coord1[0] - coord0[0], coord1[1] - coord0[1]
        x2, y2 = coords[:, 0] - coord0[0], coords[:, 1] - coord0[1]
        return np.where(np.abs(x1 * y2 - x2 * y1) < 1e-14)[0]
    for tags in constraints_vertices_tags:
        constraints.append(domain.create_region('Constraints', "vertices by get_edge", 'facet', functions={'get_edge' : Function('get_edge', get_edge, extra_args={'tags': tags})}, allow_empty=True))

    omega = domain.create_region('Omega', 'all')
    
    field = Field.from_args('fu', np.float64, 'vector', omega, approx_order=1)
    # field = Field.from_args('fu', np.float64, 'vector', omega, approx_order=2)

    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')

    m = Material('m', D=stiffness_from_lame(dim=2, lam=1.0, mu=1.0))
    # m = Material('m', D=stiffness_from_youngpoisson(dim = 2, young=207, poisson=0.3))
    f = Material('f', values={".val": [-float(force[0]), -float(force[1])]})

    integral = Integral('i', order=2)
    integral_force = Integral('if', order=0)

    t1 = Term.new('dw_lin_elastic(m.D, v, u)',
              integral, omega, m=m, v=v, u=u)
    t2 = Term.new('dw_point_load(f.val, v)',
                integral_force, forces, f=f, v=v)
    
    eq = Equation('balance', t1 + t2)
    eqs = Equations([eq])
    fix_u_list = [EssentialBC('fix_u', constraint, {'u.all' : 0.0}) for constraint in constraints]
    # shift_u = EssentialBC('shift_u', gamma2, {'u.0' : bc_fun})
    # fix_u = EssentialBC('fix_u', constraints, {'u.all' : 0.0})
    # def shift_u_fun(ts, coors, bc=None, problem=None, shift=0.0):
    #                 val = shift
    #                 return val
    # bc_fun = Function('shift_u_fun', shift_u_fun,
    #                 extra_args={'shift' : 0.01})
    # shift_u = EssentialBC('shift_u', forces, {'u.0' : bc_fun})
    
    ls = ScipyDirect({})
    nls_status = IndexedStruct()
    nls = Newton({}, lin_solver=ls, status=nls_status)

    pb = Problem('elasticity', equations=eqs)
    pb.save_regions_as_groups('regions')
    pb.set_bcs(ebcs=Conditions(fix_u_list))
    pb.set_solver(nls)
    status = IndexedStruct()
    variables = pb.solve(status=status)
    print('Nonlinear solver status:\n', nls_status)
    print('Stationary solver status:\n', status)
    pb.save_state('linear_elasticity.vtk', variables)