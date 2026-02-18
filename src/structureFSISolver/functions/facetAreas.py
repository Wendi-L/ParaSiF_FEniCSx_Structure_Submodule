"""
##############################################################################
# Parallel Partitioned Multi-Physics Simulation Framework (ParaSiF)          #
#                                                                            #
# Copyright (C) 2025 The ParaSiF Development Team                            #
# All rights reserved                                                        #
#                                                                            #
# This software is licensed under the GNU General Public License version 3   #
#                                                                            #
# ** GNU General Public License, version 3 **                                #
#                                                                            #
# This program is free software: you can redistribute it and/or modify       #
# it under the terms of the GNU General Public License as published by       #
# the Free Software Foundation, either version 3 of the License, or          #
# (at your option) any later version.                                        #
#                                                                            #
# This program is distributed in the hope that it will be useful,            #
# but WITHOUT ANY WARRANTY; without even the implied warranty of             #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
# GNU General Public License for more details.                               #
#                                                                            #
# You should have received a copy of the GNU General Public License          #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.      #
##############################################################################

    @file facetAreas.py
    
    @author W. Liu
    
    @brief facet Areas file of the structure code.

"""

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________
from dolfinx import *
from dolfinx.cpp.mesh import entities_to_geometry
import ufl
from ufl import (FacetArea)
import numpy as np
from mpi4py import MPI
import math
import scipy as sp
from scipy import spatial as sp_spatial
from scipy.spatial import Delaunay, ConvexHull

class facetAreas:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define facet areas
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def calculate_area(self, p1, p2, p3):
        # Heron's formula to calculate the area of a triangle
        a = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**0.5
        b = ((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)**0.5
        c = ((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)**0.5
    
        s = 0.5 * (a + b + c)
        area = (s * (s - a) * (s - b) * (s - c))**0.5
        return area

    def calculate_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def find_smallest_distance(self, points):
        num_points = len(points)
        
        # Initialize the minimum distance to a large value
        min_distance = float('inf')
        
        # Iterate through all pairs of points
        for i in range(num_points - 1):
            for j in range(i + 1, num_points):
                distance = self.calculate_distance(points[i], points[j])
                min_distance = min(min_distance, distance)
    
        return min_distance

    def facets_area_list_calculation(self,
                                     domain,
                                     FunctionSpace,
                                     dofs_fetch_list,
                                     dimension):
        comm = domain.comm
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)
        domain.topology.create_connectivity(tdim, fdim)
        # Ensure connectivity exists
        domain.topology.create_connectivity(fdim, tdim)
        # Get locally owned exterior facets
        facets = mesh.exterior_facet_indices(domain.topology)
        imap = domain.topology.index_map(fdim)
        facets = facets[facets < imap.size_local]
        # Call once
        facet_dofs = fem.locate_dofs_topological(
            FunctionSpace, fdim, facets
        )
        # Map facets to geometry (vertex indices)
        geom_dofs = entities_to_geometry(domain._cpp_object, fdim, facets, False)
        geom = domain.geometry.x
        
        for facet, vertices in zip(facets, geom_dofs):
            coords = geom[vertices]
            if dimension == 2:
                # Edge length
                pa, pb = coords
                area = np.linalg.norm(pb - pa)

            elif dimension == 3:
                # Triangle area
                pa, pb, pc = coords
                area = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa))

            else:
                raise RuntimeError("Unsupported dimension")

            ndofs = len(facet_dofs)
            if ndofs > 0:
                area_per_dof = area / ndofs
            else:
                continue

            for dof in facet_dofs:
                if dof in dofs_fetch_list:
                    self.areaf_vec[dof] += area_per_dof

        # Synchronise all ranks
        domain.comm.Barrier()
        # Finalise parallel accumulation
        # Compute total area safely
        local_sum = 0.0

        for iii, ppp in enumerate(self.areaf_vec):
            local_sum += self.areaf_vec[iii]
        areatotal = comm.allreduce(local_sum, op=MPI.SUM)

        if self.rank == 0 and self.iDebug():
            print("Total area of MUI fetched surface= ", areatotal, " m^2")

    def facets_area_define(self,
                           mesh,
                           Q,
                           dofs_fetch_list,
                           gdim):
            # Define function for facet area
            self.areaf= fem.Function(Q)
            #self.areaf_vec = self.areaf.vector().get_local()
            self.areaf_vec = self.areaf.x.array

            if self.iLoadAreaList():
                # hdf5meshAreaDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/mesh_boundary_and_values.h5", "r")
                # hdf5meshAreaDataInTemp.read(self.areaf, "/areaf/vector_0")
                # hdf5meshAreaDataInTemp.close()
                pass
            else:
                if self.rank == 0: print ("{FENICS} facet area calculating")
                # Calculate function for facet area
                self.facets_area_list_calculation(mesh, Q, dofs_fetch_list, gdim)
                # Apply the facet area vectors
                # self.areaf.vector().set_local(self.areaf_vec)
                # self.areaf.vector().apply("insert")
                self.areaf.x.array[:] = self.areaf_vec
                self.areaf.x.scatter_forward()
                # Facet area vectors I/O
                # if (self.iHDF5FileExport()) and (self.iHDF5MeshExport()):
                #     hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "a")
                #     hdfOutTemp.write(self.areaf, "/areaf")
                #     hdfOutTemp.close()
                # else:
                #     pass

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#