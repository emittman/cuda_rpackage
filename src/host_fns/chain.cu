#include "../header/chain.h"
#include "../header/iter_getter.h"

data_t::data_t(double* _yty, double* _xty, double* _ytx, double* _xtx, int _G, int _V, int _N): 
    yty(_yty, _yty + _G), xty(_xty, _xty + _G*_V), ytx(_ytx, _ytx+_G*_V), xtx(_xtx, _xtx + _V*_V), G(_G), V(_V), N(_N) {}
