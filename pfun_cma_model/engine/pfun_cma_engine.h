/*
  pfun_cma_engine.h: Header definitions for the PFun CMA Model Engine
*/

// Low-level numerical methods

float exp(float x);

float expit_pfun(float x);

float calc_vdep_current(float v, float v1, float v2, float A=1.0, float B=1.0);

float E_norm(float x);

float normalize(x, float a = 0.0, float b = 1.0);
