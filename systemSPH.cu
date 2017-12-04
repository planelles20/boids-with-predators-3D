//-----------------------------------------------------------------------
// Code file of SistemSPH class which simulate boids model using
// Smoothed-particle hydrodynamics (SPH) method.
//
// Licensing: This code is distributed under the Apache License 2.0
// Author: Carlos Planelles Alemany, planelles20(at)gmail(dot)com
//-----------------------------------------------------------------------

#include "systemSPH.h"

////////////////////////////////  kernels  ////////////////////////////////////

__global__ void clearGridIndices_kernel(intvec2 *d_inidices, int numCells) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < numCells){
        d_inidices[i].init = 0;
        d_inidices[i].end  = 0;
    }
}

__global__ void builtGridIncices_kernel(intvec2 *gridIdx, particle *d_particle, int numParticles){
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < numParticles){
        if( i==0 && d_particle[i].cellIdx != d_particle[i+1].cellIdx){
            gridIdx[d_particle[i].cellIdx].init = 0;
        }

        if(i!=0 && d_particle[i].cellIdx != d_particle[i-1].cellIdx){
            gridIdx[d_particle[i].cellIdx].init = i;
        }
        if(i!=(numParticles-1) && d_particle[i].cellIdx != d_particle[i+1].cellIdx){
            gridIdx[d_particle[i].cellIdx].end = i;
        }
        if(i==(numParticles-1) && d_particle[i].cellIdx == d_particle[i-1].cellIdx){
            gridIdx[d_particle[i].cellIdx].end = i;
        }
    }
}

__global__ void posParticleCell_kernel(particle *d_particle, int numParticles, int xMesh, int yMesh, int zMesh){
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<numParticles){
        int ii = int(d_particle[i].x*float(xMesh));
        int jj = int(d_particle[i].y*float(yMesh));
        int kk = int(d_particle[i].z*float(zMesh));
        if(ii == xMesh) ii--;
        if(jj == yMesh) jj--;
        if(kk == zMesh) kk--;
        d_particle[i].cellIdx = ii+jj*xMesh+kk*xMesh*yMesh;
    }
}

__global__ void bitonic_sort_kernel(particle *dev_values, int j, int k) {
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i].cellIdx>dev_values[ixj].cellIdx) {
        /* exchange(i,ixj) particles; */
        //cell
        particle temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i].cellIdx<dev_values[ixj].cellIdx) {
        /* exchange(i,ixj); */
        particle temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

__global__ void calcul_kernel(particle *d_particle, intvec2 *d_inidices,
                              calcStruct *d_calc, int numParticles,
                              int xMesh, int yMesh, int zMesh){

    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < numParticles){
        // number of near boids
        int sepa_n = 0;
        int alig_n = 0;
        int cohe_n = 0;
        int pred_n = 0;

        float sepa_x = 0.0,  sepa_y = 0.0,  sepa_z = 0.0;
        float alig_x = 0.0,  alig_y = 0.0,  alig_z = 0.0;
        float cohe_x = 0.0,  cohe_y = 0.0,  cohe_z = 0.0;
        float pred_x = 0.0,  pred_y = 0.0,  pred_z = 0.0;

        float cohe_r = 1.0/float(xMesh);
        float alig_r = 0.7*cohe_r;
        float sepa_r = 0.4*cohe_r;

        if(d_particle[i].tag > 0){
            alig_r = 0.2*cohe_r;
            sepa_r = 0.1*cohe_r;
        }

        // look own cell
        for(int ii=d_inidices[d_particle[i].cellIdx].init; ii<d_inidices[d_particle[i].cellIdx].end; ++ii){
            float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                              pow(d_particle[ii].y-d_particle[i].y, 2)+
                              pow(d_particle[ii].z-d_particle[i].z, 2));

            if(d_particle[ii].tag > 0 && d_particle[i].tag < 1 && i!=ii){
                pred_n++;
                pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
            }
            else if(dist < sepa_r && i!=ii){
               sepa_n++;
               sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
               sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
               sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
            }
            else if(dist < alig_r && i!=ii){
               alig_n++;
               alig_x += d_particle[ii].vx;
               alig_y += d_particle[ii].vy;
               alig_z += d_particle[ii].vz;
            }
            else if(dist < cohe_r && i!=ii){
               cohe_n++;
               cohe_x += d_particle[ii].x;
               cohe_y += d_particle[ii].y;
               cohe_z += d_particle[ii].z;
            }
        }
        /// looking near cells
        // x+1
        if(d_particle[i].cellIdx+1 < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1].init; ii<d_inidices[d_particle[i].cellIdx+1].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x-1
        if(d_particle[i].cellIdx-1 > 0){
            for(int ii=d_inidices[d_particle[i].cellIdx-1].init; ii<d_inidices[d_particle[i].cellIdx-1].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // y+1
        if(d_particle[i].cellIdx+1*xMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1*xMesh].init; ii<d_inidices[d_particle[i].cellIdx+1*xMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // y-1
        if(d_particle[i].cellIdx-1*xMesh > 0){
            for(int ii=d_inidices[d_particle[i].cellIdx-1*xMesh].init; ii<d_inidices[d_particle[i].cellIdx-1*xMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // z+1
        if(d_particle[i].cellIdx+1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx+1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // z-1
        if(d_particle[i].cellIdx-1*xMesh*yMesh > 0){
            for(int ii=d_inidices[d_particle[i].cellIdx-1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx-1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x+1, y+1
        if(d_particle[i].cellIdx+1+1*xMesh > 0 && d_particle[i].cellIdx+1+1*xMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1+1*xMesh].init; ii<d_inidices[d_particle[i].cellIdx+1+1*xMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x+1, y-1
        if(d_particle[i].cellIdx+1-1*xMesh > 0 && d_particle[i].cellIdx+1-1*xMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1-1*xMesh].init; ii<d_inidices[d_particle[i].cellIdx+1-1*xMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x-1, y+1
        if(d_particle[i].cellIdx-1+1*xMesh > 0 && d_particle[i].cellIdx-1+1*xMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx-1+1*xMesh].init; ii<d_inidices[d_particle[i].cellIdx-1+1*xMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x-1, y-1
        if(d_particle[i].cellIdx-1-1*xMesh > 0 && d_particle[i].cellIdx-1-1*xMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx-1-1*xMesh].init; ii<d_inidices[d_particle[i].cellIdx-1-1*xMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x+1, y+1, z+1
        if(d_particle[i].cellIdx+1+1*xMesh+1*xMesh*yMesh > 0 && d_particle[i].cellIdx+1+1*xMesh+1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1+1*xMesh+1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx+1+1*xMesh+1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x+1, y-1, z+1
        if(d_particle[i].cellIdx+1-1*xMesh+1*xMesh*yMesh > 0 && d_particle[i].cellIdx+1-1*xMesh+1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1-1*xMesh+1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx+1-1*xMesh+1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x-1, y+1, z+1
        if(d_particle[i].cellIdx-1+1*xMesh+1*xMesh*yMesh > 0 && d_particle[i].cellIdx-1+1*xMesh+1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx-1+1*xMesh+1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx-1+1*xMesh+1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x-1, y-1, z+1
        if(d_particle[i].cellIdx-1-1*xMesh+1*xMesh*yMesh > 0 && d_particle[i].cellIdx-1-1*xMesh+1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx-1-1*xMesh+1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx-1-1*xMesh+1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x+1, y+1, z-1
        if(d_particle[i].cellIdx+1+1*xMesh-1*xMesh*yMesh > 0 && d_particle[i].cellIdx+1+1*xMesh-1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1+1*xMesh-1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx+1+1*xMesh-1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x+1, y-1, z-1
        if(d_particle[i].cellIdx+1-1*xMesh-1*xMesh*yMesh > 0 && d_particle[i].cellIdx+1-1*xMesh-1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1-1*xMesh-1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx+1-1*xMesh-1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x-1, y+1, z-1
        if(d_particle[i].cellIdx-1+1*xMesh-1*xMesh*yMesh > 0 && d_particle[i].cellIdx-1+1*xMesh-1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx-1+1*xMesh-1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx-1+1*xMesh-1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x-1, y-1, z-1
        if(d_particle[i].cellIdx-1-1*xMesh-1*xMesh*yMesh > 0 && d_particle[i].cellIdx-1-1*xMesh-1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx-1-1*xMesh-1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx-1-1*xMesh-1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x+1, y, z+1
        if(d_particle[i].cellIdx+1+1*xMesh*yMesh > 0 && d_particle[i].cellIdx+1+1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1+1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx+1+1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x-1, y, z+1
        if(d_particle[i].cellIdx-1+1*xMesh*yMesh > 0 && d_particle[i].cellIdx-1+1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx-1+1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx-1+1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x, y+1, z+1
        if(d_particle[i].cellIdx+1*xMesh+1*xMesh*yMesh > 0 && d_particle[i].cellIdx+1*xMesh+1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1*xMesh+1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx+1*xMesh+1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x, y-1, z+1
        if(d_particle[i].cellIdx-1*xMesh+1*xMesh*yMesh > 0 && d_particle[i].cellIdx-1*xMesh+1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx-1*xMesh+1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx-1*xMesh+1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x+1, y, z-1
        if(d_particle[i].cellIdx+1-1*xMesh*yMesh > 0 && d_particle[i].cellIdx+1-1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1-1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx+1-1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x-1, y, z-1
        if(d_particle[i].cellIdx-1-1*xMesh*yMesh > 0 && d_particle[i].cellIdx-1-1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx-1-1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx-1-1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x, y+1, z-1
        if(d_particle[i].cellIdx+1*xMesh-1*xMesh*yMesh > 0 && d_particle[i].cellIdx+1*xMesh-1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx+1*xMesh-1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx+1*xMesh-1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }
        // x, y-1, z-1
        if(d_particle[i].cellIdx-1*xMesh-1*xMesh*yMesh > 0 && d_particle[i].cellIdx-1*xMesh-1*xMesh*yMesh < xMesh*yMesh*zMesh){
            for(int ii=d_inidices[d_particle[i].cellIdx-1*xMesh-1*xMesh*yMesh].init; ii<d_inidices[d_particle[i].cellIdx-1*xMesh-1*xMesh*yMesh].end; ++ii){
                float dist = sqrt(pow(d_particle[ii].x-d_particle[i].x, 2)+
                                  pow(d_particle[ii].y-d_particle[i].y, 2)+
                                  pow(d_particle[ii].z-d_particle[i].z, 2));

                if(d_particle[ii].tag > 0 && d_particle[i].tag < 1){
                    pred_n++;
                    pred_x += (d_particle[i].x-d_particle[ii].x)/dist;
                    pred_y += (d_particle[i].y-d_particle[ii].y)/dist;
                    pred_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < sepa_r){
                   sepa_n++;
                   sepa_x += (d_particle[i].x-d_particle[ii].x)/dist;
                   sepa_y += (d_particle[i].y-d_particle[ii].y)/dist;
                   sepa_z += (d_particle[i].z-d_particle[ii].z)/dist;
                }
                else if(dist < alig_r){
                   alig_n++;
                   alig_x += d_particle[ii].vx;
                   alig_y += d_particle[ii].vy;
                   alig_z += d_particle[ii].vz;
                }
                else if(dist < cohe_r){
                   cohe_n++;
                   cohe_x += d_particle[ii].x;
                   cohe_y += d_particle[ii].y;
                   cohe_z += d_particle[ii].z;
                }
            }
        }

        //Predator
        if(sepa_n > 0){
            float mod = sqrt(pow(sepa_x, 2)+pow(sepa_y, 2)+pow(sepa_z, 2));
            pred_x = pred_x/mod-d_particle[i].vx;
            pred_y = pred_y/mod-d_particle[i].vy;
            pred_z = pred_z/mod-d_particle[i].vz;

            d_calc[i].pred.x = pred_x;
            d_calc[i].pred.y = pred_y;
            d_calc[i].pred.z = pred_z;
        }
        else {
            d_calc[i].pred.x = 0.0;
            d_calc[i].pred.y = 0.0;
            d_calc[i].pred.z = 0.0;
        }

        //Separation
        if(sepa_n > 0){
            float mod = sqrt(pow(sepa_x, 2)+pow(sepa_y, 2)+pow(sepa_z, 2));
            sepa_x = sepa_x/mod-d_particle[i].vx;
            sepa_y = sepa_y/mod-d_particle[i].vy;
            sepa_z = sepa_z/mod-d_particle[i].vz;


            d_calc[i].sepa.x = sepa_x;
            d_calc[i].sepa.y = sepa_y;
            d_calc[i].sepa.z = sepa_z;
        }
        else {
            d_calc[i].sepa.x = 0.0;
            d_calc[i].sepa.y = 0.0;
            d_calc[i].sepa.z = 0.0;
        }

        // Alignment
        if(alig_n > 0){
            float mod = sqrt(pow(alig_x, 2)+pow(alig_y, 2)+pow(alig_z, 2));
            alig_x = alig_x/mod-d_particle[i].vx;
            alig_y = alig_y/mod-d_particle[i].vy;
            alig_z = alig_z/mod-d_particle[i].vz;

            d_calc[i].alig.x = alig_x;
            d_calc[i].alig.y = alig_y;
            d_calc[i].alig.z = alig_z;
        }
        else {
            d_calc[i].alig.x = 0.0;
            d_calc[i].alig.y = 0.0;
            d_calc[i].alig.z = 0.0;
        }

        // Cohesion
        if(cohe_n > 0){
            cohe_x = cohe_x/cohe_n-d_particle[i].x;
            cohe_y = cohe_y/cohe_n-d_particle[i].y;
            cohe_z = cohe_z/cohe_n-d_particle[i].z;

            d_calc[i].cohe.x = cohe_x/sqrt(pow(cohe_x, 2)+pow(cohe_y, 2)+pow(cohe_z, 2));
            d_calc[i].cohe.y = cohe_y/sqrt(pow(cohe_x, 2)+pow(cohe_y, 2)+pow(cohe_z, 2));
            d_calc[i].cohe.z = cohe_z/sqrt(pow(cohe_x, 2)+pow(cohe_y, 2)+pow(cohe_z, 2));
        }
        else {
            d_calc[i].cohe.x = 0.0;
            d_calc[i].cohe.y = 0.0;
            d_calc[i].cohe.z = 0.0;
        }


    }
}

__global__ void integrate_kernel(particle *d_particle, calcStruct *d_calc, int numParticles, float seed){
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < numParticles){

        curandState_t state;
        curand_init(seed,
                    i,
                    0,
                    &state);

        float dt =0.002;
        // dv
        //d_particle[i].theta = d_calc[i].thetaMedium + 0.1*curand_normal(&state);
        //d_particle[i].alpha = d_calc[i].alphaMedium + 0.1*curand_normal(&state);

        float A = 10.0, B = 10.0, C = 10.0, D = 50.0;

        float dvx = A*d_calc[i].sepa.x+B*d_calc[i].alig.x+C*d_calc[i].cohe.x+D*d_calc[i].pred.x;
        float dvy = A*d_calc[i].sepa.y+B*d_calc[i].alig.y+C*d_calc[i].cohe.y+D*d_calc[i].pred.y;
        float dvz = A*d_calc[i].sepa.z+B*d_calc[i].alig.z+C*d_calc[i].cohe.z+D*d_calc[i].pred.z;

        // v
        float vx = d_particle[i].vx + dvx*dt;
        float vy = d_particle[i].vy + dvy*dt;
        float vz = d_particle[i].vz + dvz*dt;

        d_particle[i].vx = vx;
        d_particle[i].vy = vy;
        d_particle[i].vz = vz;

        // x
        if(d_particle[i].x<0.0 && vx<0.0) {
            d_particle[i].x = 1.0;
            d_particle[i].y = d_particle[i].y+vy*dt;
            d_particle[i].z = d_particle[i].z+vz*dt;
        }
        else if(d_particle[i].x>1.0 && vx > 0.0) {
            //theta = pi-theta;
            d_particle[i].x = 0.0;
            d_particle[i].y = d_particle[i].y+vy*dt;
            d_particle[i].z = d_particle[i].z+vz*dt;
        }
        else if(d_particle[i].y<0.0 && vy < 0.0) {
            //theta = 2.0*pi-theta;
            d_particle[i].x = d_particle[i].x+vx*dt;
            d_particle[i].y = 1.0;
            d_particle[i].z = d_particle[i].z+vz*dt;
        }
        else if(d_particle[i].y>1.0 && vy > 0.0) {
            //theta = 2.0*pi-theta;
            d_particle[i].x = d_particle[i].x+vx*dt;
            d_particle[i].y = 0.0;
            d_particle[i].z = d_particle[i].z+vz*dt;
        }
        else if(d_particle[i].z<0.0 && vz < 0.0) {
            //theta = 2.0*pi-theta;
            d_particle[i].x = d_particle[i].x+vx*dt;
            d_particle[i].y = d_particle[i].y+vy*dt;
            d_particle[i].z = 1.0;
        }
        else if(d_particle[i].z>1.0 && vz > 0.0) {
            d_particle[i].x = d_particle[i].x+vx*dt;
            d_particle[i].y = d_particle[i].y+vy*dt;
            d_particle[i].z = 0.0;
        }
        else {
            d_particle[i].x = d_particle[i].x+vx*dt;
            d_particle[i].y = d_particle[i].y+vy*dt;
            d_particle[i].z = d_particle[i].z+vz*dt;
        }
    }
}

/////////////////////////////  methods ////////////////////////////////////////

SystemSPH::SystemSPH(unsigned int blocks,
                     unsigned int threads,
                     unsigned int xMesh,
                     unsigned int yMesh,
                     unsigned int zMesh) {

    //glewInit();
    //glewExperimental = GL_TRUE;
    this->numBlocks = blocks;
    this->numThreads = threads;
    this->numParticles = blocks*threads;
    this->xMeshDim = xMesh;
    this->yMeshDim = yMesh;
    this->zMeshDim = zMesh;
    this->numIndices = xMesh*yMesh*zMesh;

    this->h_particle = new particle[this->numParticles];
    this->h_inidices = new intvec2[this->numIndices];
    this->h_calc = new calcStruct[this->numParticles];

    this->particleIndices = new GLushort[this->numParticles];

    //init particle values
    this->InitParticleData();

    //Generate openGL buffers
    //vertex array object
    //Create vertex buffer object(s)+
    glGenVertexArrays(1, &this->VAO);
    //Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
    glBindVertexArray(this->VAO);


    // set vertex buffer
    glGenBuffers(1, &this->VBOparticles);
    glBindBuffer(GL_ARRAY_BUFFER, this->VBOparticles);
    glBufferData(GL_ARRAY_BUFFER, this->numParticles*sizeof(particle), this->h_particle, GL_DYNAMIC_COPY); //like cupdamemcy host->device
    cudaGraphicsGLRegisterBuffer(&this->cudaResourceBufParticles, this->VBOparticles, cudaGraphicsRegisterFlagsNone);

    glGenBuffers(1, &this->EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->numParticles*sizeof(GLushort), this->particleIndices, GL_STATIC_DRAW);

    // bind attribute pointer(s)
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(particle), (GLvoid*)(2 * sizeof(GLint)));
    glEnableVertexAttribArray(0);
    // Velocity attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(particle), (GLvoid*)(2*sizeof(GLint)+3*sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    // tag attribute
    glVertexAttribPointer(2, 1, GL_INT, GL_FALSE, sizeof(particle), (GLvoid*)(2*sizeof(GLint)+6*sizeof(GLfloat)));
    glEnableVertexAttribArray(2);

    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); //unbind EBO
    glBindBuffer(GL_ARRAY_BUFFER, 0); // unbind VBO
    glBindVertexArray(0); // Unbind VAO

    glGenBuffers(1, &this->VBOindices);
    glBindBuffer(GL_ARRAY_BUFFER, this->VBOindices);
    glBufferData(GL_ARRAY_BUFFER, this->numIndices*sizeof(intvec2), this->h_inidices, GL_DYNAMIC_COPY); //like cupdamemcy host->device
    cudaGraphicsGLRegisterBuffer(&this->cudaResourceBufIndices, this->VBOindices, cudaGraphicsRegisterFlagsNone);


    glGenBuffers(1, &this->VBOcalc);
    glBindBuffer(GL_ARRAY_BUFFER, this->VBOcalc);
    glBufferData(GL_ARRAY_BUFFER, this->numParticles*sizeof(calcStruct), this->h_calc, GL_DYNAMIC_COPY); //like cupdamemcy host->device
    cudaGraphicsGLRegisterBuffer(&this->cudaResourceBufCalc, this->VBOcalc, cudaGraphicsRegisterFlagsNone);

    /// boundaries
    // allocate memory
    this->boundaryPoints = new point[8];
    this->boundaryIndices = new GLushort[16]; // points per squares dot num of squares
    //create boundaries (init)
    this->CreateBounderiesPoints();
    this->CreateBounderiesIndices();

    //vertex array object
    glGenVertexArrays(1, &this->VAOboundary);
    glBindVertexArray(this->VAOboundary);


    //Create vertex buffer object
    glGenBuffers(1, &this->VBOboundary);
    glBindBuffer(GL_ARRAY_BUFFER, this->VBOboundary);
    glBufferData(GL_ARRAY_BUFFER, sizeof(*this->boundaryPoints)*8, this->boundaryPoints, GL_STATIC_DRAW);

    //Create Element Buffer Objects
    glGenBuffers(1, &this->EBOboundary);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBOboundary);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(*this->boundaryIndices)*16, this->boundaryIndices, GL_STATIC_DRAW);

    // Position attribute (3D)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(*this->boundaryPoints), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0); // unbind VBOp
    glBindVertexArray(0); // Unbind VAOp
}

SystemSPH::~SystemSPH() {
    //cudaFree(d_particle);
    //cudaFree(d_inidices);
    glDeleteVertexArrays(1, &this->VAO);
    glDeleteBuffers(1, &this->VBOparticles);
    glDeleteBuffers(1, &this->VBOindices);
    glDeleteBuffers(1, &this->VBOcalc);
    glDeleteBuffers(1, &this->EBO);

    cudaGraphicsUnregisterResource(this->cudaResourceBufParticles);
    cudaGraphicsUnregisterResource(this->cudaResourceBufIndices);
    cudaGraphicsUnregisterResource(this->cudaResourceBufCalc);
}

void SystemSPH::Particle_print() {
    //copy to host
    cudaGraphicsMapResources(1, &this->cudaResourceBufParticles, 0);
    size_t size = this->numParticles*sizeof(particle);
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_particle, &size, this->cudaResourceBufParticles);
    std::cout << cudaMemcpy(this->h_particle, this->d_particle, this->numParticles*sizeof(particle), cudaMemcpyDeviceToHost);
    cudaGraphicsUnmapResources(1, &this->cudaResourceBufParticles, 0);

    for (int i = 0; i < this->numParticles; ++i) {
        printf("Particle id: %d, particle cell id: %d, position: (%1.3f, %1.3f, %1.3f), velocity:  (%1.3f, %1.3f, %1.3f),",
                this->h_particle[i].id, this->h_particle[i].cellIdx,
                this->h_particle[i].x, this->h_particle[i].y, this->h_particle[i].z,
                this->h_particle[i].vx, this->h_particle[i].vy, this->h_particle[i].vz);
        printf("\n");
    }
}

void SystemSPH::Indices_print() {
    //copy to host
    cudaGraphicsMapResources(1, &this->cudaResourceBufIndices, 0);
    size_t size = this->numIndices*sizeof(intvec2);
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_inidices, &size, this->cudaResourceBufIndices);
    cudaMemcpy(this->h_inidices, this->d_inidices, this->numIndices*sizeof(intvec2), cudaMemcpyDeviceToHost);
    cudaGraphicsUnmapResources(1, &this->cudaResourceBufIndices, 0);
    //print indices
    for (int i = 0; i < this->numIndices; ++i) {
      printf("Indice pos: %d, init: %d, end: %d, number of particles: %d",
              i, this->h_inidices[i].init, this->h_inidices[i].end,
              this->h_inidices[i].end-this->h_inidices[i].init);
      printf("\n");
    }
}

void SystemSPH::Calc_print() {
    //copy to host
    cudaGraphicsMapResources(1, &this->cudaResourceBufCalc, 0);
    size_t size = this->numParticles*sizeof(calcStruct);
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_calc, &size, this->cudaResourceBufCalc);
    std::cout << cudaMemcpy(this->h_calc, this->d_calc, this->numParticles*sizeof(calcStruct), cudaMemcpyDeviceToHost);
    cudaGraphicsUnmapResources(1, &this->cudaResourceBufCalc, 0);
    //print indices
    for (int i = 0; i < this->numParticles; ++i) {
      printf("Particle index: %d. Separation = (%1.3f, %1.3f., %1.3f).",
              i, this->h_calc[i].sepa.x, this->h_calc[i].sepa.y, this->h_calc[i].sepa.z);
      printf("\n");
    }
}

void SystemSPH::InitParticleData(){
    srand(time(NULL));
    for (int i = 0; i < this->numParticles; ++i) {
        this->particleIndices[i] = i;
        this->h_particle[i].id = i;
        this->h_particle[i].x = (float)rand()/(float)RAND_MAX;
        this->h_particle[i].y = (float)rand()/(float)RAND_MAX;
        this->h_particle[i].z = (float)rand()/(float)RAND_MAX;
        this->h_particle[i].vx = 1.0*(2.0*(float)rand()/(float)RAND_MAX-1.0);
        this->h_particle[i].vy = 1.0*(2.0*(float)rand()/(float)RAND_MAX-1.0);
        this->h_particle[i].vz = 1.0*(2.0*(float)rand()/(float)RAND_MAX-1.0);
        if(i<25)
            this->h_particle[i].tag = 1;
        else
            this->h_particle[i].tag = 0;
    }
}

void SystemSPH::SortParticles(){

    //get device direction
    //modificate vertex buffers with cuda
    cudaGraphicsMapResources(1, &this->cudaResourceBufParticles, 0);
    size_t size = sizeof(particle)*this->numParticles;
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_particle, &size, this->cudaResourceBufParticles);

    dim3 blocks(this->numBlocks,1,1);    /* Number of blocks   */
    dim3 threads(this->numThreads,1,1);  /* Number of threads  */

    int j, k;
    /* Major step */
    for (k = 2; k <= this->numParticles; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
                bitonic_sort_kernel<<<blocks, threads>>>(this->d_particle, j, k);
        }
    }
    cudaGraphicsUnmapResources(1, &this->cudaResourceBufParticles, 0);
}

void SystemSPH::ClearGridIndices(){

    //get device direction
    //modificate vertex buffers with cuda
    cudaGraphicsMapResources(1, &this->cudaResourceBufIndices, 0);
    size_t size = sizeof(intvec2)*this->numIndices;
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_inidices, &size, this->cudaResourceBufIndices);

    dim3 blocks(this->numIndices%1024==0
         ? int(this->numIndices/1024) : int(this->numIndices/1024)+1 ,1 ,1);    /* Number of blocks   */
    dim3 threads(1024,1,1);  /* Number of threads  */
    clearGridIndices_kernel<<<blocks, threads>>>(this->d_inidices, this->numParticles);

    cudaGraphicsUnmapResources(1, &this->cudaResourceBufIndices, 0);

}

void SystemSPH::BuiltGridIncices(){
    //get device direction
    //Indices
    cudaGraphicsMapResources(1, &this->cudaResourceBufIndices, 0);
    size_t sizeIndices = sizeof(intvec2)*this->numIndices;
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_inidices, &sizeIndices, this->cudaResourceBufIndices);
    //Paricles
    cudaGraphicsMapResources(1, &this->cudaResourceBufParticles, 0);
    size_t sizeParticle = sizeof(particle)*this->numParticles;
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_particle, &sizeParticle, this->cudaResourceBufParticles);

    dim3 blocks(this->numBlocks,1,1);    /* Number of blocks   */
    dim3 threads(this->numThreads,1,1);  /* Number of threads  */
    builtGridIncices_kernel<<<blocks, threads>>>(this->d_inidices, this->d_particle, this->numParticles);

    cudaGraphicsUnmapResources(1, &this->cudaResourceBufIndices, 0);
    cudaGraphicsUnmapResources(1, &this->cudaResourceBufParticles, 0);
}

void SystemSPH::CreateGridIndices(){
    this->ClearGridIndices();
    this->BuiltGridIncices();
}

void SystemSPH::PosParticleCell(){
    //get device direction
    cudaGraphicsMapResources(1, &this->cudaResourceBufParticles, 0);
    size_t sizeParticle = sizeof(particle)*this->numParticles;
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_particle, &sizeParticle, this->cudaResourceBufParticles);

    dim3 blocks(this->numBlocks,1,1);    /* Number of blocks   */
    dim3 threads(this->numThreads,1,1);  /* Number of threads  */
    posParticleCell_kernel<<<blocks, threads>>>(this->d_particle, this->numParticles,
                                                this->xMeshDim, this->yMeshDim, this->zMeshDim);

    cudaGraphicsUnmapResources(1, &this->cudaResourceBufParticles, 0);
}

void SystemSPH::CalcOperations(){
    //get device direction
    //Indices
    cudaGraphicsMapResources(1, &this->cudaResourceBufIndices, 0);
    size_t sizeIndices = sizeof(intvec2)*this->numIndices;
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_inidices, &sizeIndices, this->cudaResourceBufIndices);
    //Paricles
    cudaGraphicsMapResources(1, &this->cudaResourceBufParticles, 0);
    size_t sizeParticle = sizeof(particle)*this->numParticles;
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_particle, &sizeParticle, this->cudaResourceBufParticles);
    //calculation
    cudaGraphicsMapResources(1, &this->cudaResourceBufCalc, 0);
    size_t sizeCalc = sizeof(calcStruct)*this->numParticles;
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_calc, &sizeCalc, this->cudaResourceBufCalc);

    dim3 blocks(this->numBlocks,1,1);    /* Number of blocks   */
    dim3 threads(this->numThreads,1,1);  /* Number of threads  */
    // calculate operations
    calcul_kernel<<<blocks, threads>>>(this->d_particle, this->d_inidices, this->d_calc, this->numParticles,
                                       this->xMeshDim, this->yMeshDim, this->zMeshDim);
    // integrate (Euler Exlicit)
    integrate_kernel<<<blocks, threads>>>(this->d_particle, this->d_calc, this->numParticles, this->seed);

    cudaGraphicsUnmapResources(1, &this->cudaResourceBufIndices, 0);
    cudaGraphicsUnmapResources(1, &this->cudaResourceBufParticles, 0);
    cudaGraphicsUnmapResources(1, &this->cudaResourceBufCalc, 0);
}

void SystemSPH::Calculate(){
    //ind cell pos indes
    this->PosParticleCell();
    //sort particles
    this->SortParticles();
    //create indices
    this->CreateGridIndices();
    //calculation
    this->CalcOperations();

}

void SystemSPH::SeedUpdate(int i){
    this->seed = i;
}

void SystemSPH::Save(const std::string& nameFile){
    //Paricles
    cudaGraphicsMapResources(1, &this->cudaResourceBufParticles, 0);
    size_t size = this->numParticles*sizeof(particle);
    cudaGraphicsResourceGetMappedPointer((void **)&this->d_particle, &size, this->cudaResourceBufParticles);
    std::cout << cudaMemcpy(this->h_particle, this->d_particle, this->numParticles*sizeof(particle), cudaMemcpyDeviceToHost);
    cudaGraphicsUnmapResources(1, &this->cudaResourceBufParticles, 0);

    std::ofstream ofs (nameFile, std::ofstream::out);
    //heater
    ofs << "Step (or seed) of simulations" << "," << this->seed << "," << "Number of particles" << "," << this->numParticles << std::endl;
    ofs << "PaticleIndice" << "," << "PaticleCell" << "," << "Position_x" << "," << "Position_y" << ","<< "Position_z" << ",";
    ofs << "Velocity_vx" << "," << "Velocity_vy" << ","<< "Velocity_vz" << std::endl;
    //body
    for(int i=0; i<this->numParticles; i++){
        ofs << this->h_particle[i].id << "," << this->h_particle[i].cellIdx << ",";
        ofs << this->h_particle[i].x << "," << this->h_particle[i].y << "," << this->h_particle[i].z << ",";
        ofs << this->h_particle[i].vx << "," << this->h_particle[i].vy << "," << this->h_particle[i].vz << std::endl;
    }
    ofs.close();

    std::cout << "The current data has been saved in: " << nameFile << std::endl;
}




//////////////////////// Plot /////////////////////////////////////////////////
void SystemSPH::DrawParticles(){
    //openGl
    glBindVertexArray(this->VAO);
    //glDrawArrays(GL_POINTS, 0, this->numParticles);
    glDrawElements(GL_POINTS, this->numParticles, GL_UNSIGNED_SHORT, 0);
    glBindVertexArray(0);
}
void SystemSPH::BackGround(float r, float g, float b, float a){
    glClearColor(r, g, b, a);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
void SystemSPH::DrawBoundary(){
    //openGl
    glBindVertexArray(this->VAOboundary);
    //glDrawArrays(GL_POINTS, 0, 8);

    glDrawElements(GL_LINE_LOOP, 16, GL_UNSIGNED_SHORT,  0);
    glBindVertexArray(0);
}
void SystemSPH::PolygonMode(){
    //this call will result in wireframe polygons.
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}

void SystemSPH::CreateBounderiesIndices(){
    // boundaries are drawed using line loop, so:
    this->boundaryIndices[0] = 0;
    this->boundaryIndices[1] = 3;
    this->boundaryIndices[2] = 2;
    this->boundaryIndices[3] = 1;
    this->boundaryIndices[4] = 0;
    this->boundaryIndices[5] = 4;
    this->boundaryIndices[6] = 7;
    this->boundaryIndices[7] = 6;
    this->boundaryIndices[8] = 5;
    this->boundaryIndices[9] = 4;
    this->boundaryIndices[10] = 7;
    this->boundaryIndices[11] = 3;
    this->boundaryIndices[12] = 2;
    this->boundaryIndices[13] = 6;
    this->boundaryIndices[14] = 5;
    this->boundaryIndices[15] = 1;
}

void SystemSPH::CreateBounderiesPoints(){
    //point 0
    this->boundaryPoints[0].x = -1.0;
    this->boundaryPoints[0].y = -1.0;
    this->boundaryPoints[0].z = -1.0;
    //point 1
    this->boundaryPoints[1].x = 1.0;
    this->boundaryPoints[1].y = -1.0;
    this->boundaryPoints[1].z = -1.0;
    //point 2
    this->boundaryPoints[2].x = 1.0;
    this->boundaryPoints[2].y = 1.0;
    this->boundaryPoints[2].z = -1.0;
    //point 3
    this->boundaryPoints[3].x = -1.0;
    this->boundaryPoints[3].y = 1.0;
    this->boundaryPoints[3].z = -1.0;
    //point 4
    this->boundaryPoints[4].x = -1.0;
    this->boundaryPoints[4].y = -1.0;
    this->boundaryPoints[4].z = 1.0;
    //point 5
    this->boundaryPoints[5].x = 1.0;
    this->boundaryPoints[5].y = -1.0;
    this->boundaryPoints[5].z = 1.0;
    //point 6
    this->boundaryPoints[6].x = 1.0;
    this->boundaryPoints[6].y = 1.0;
    this->boundaryPoints[6].z = 1.0;
    //point 7
    this->boundaryPoints[7].x = -1.0;
    this->boundaryPoints[7].y = 1.0;
    this->boundaryPoints[7].z = 1.0;
}
