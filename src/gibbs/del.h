#ifndef DEL_H
#define DEL_H

__global__ void del_kernel1(chain_t *dd){
  int g = IDX, n;
  double D0[3] = {0.0, 0.0, 0.0};

  if(g >= dd->G)
    return;

  approx_gibbs_args_t args;
  args.idx = g;
  args.x0 = dd->del[g];
  args.target_type = LTARGET_BASIC;
  args.step_width = STEP_WIDTH;
  args.max_steps = MAX_STEPS;

  args.A = 0.0;
  for(n = 0; n < dd->N; ++n){
    if(dd->group[n] == 1)
      args.A -= (double) dd->counts[I(n, g)];
    else
      args.A += (double) dd->counts[I(n, g)];
  }

  args.B = 1.0/(2.0 * dd->sigDel[0] * dd->sigDel[0] * dd->xiDel[g]);
  args.C = dd->theDel[0];

  for(n = 0; n < dd->N; ++n)
    D0[dd->group[n] - 1] += exp(dd->eps[I(n, g)]);

  args.D = exp(dd->phi[g] - dd->alp[g]) * D0[1]
         + exp(dd->phi[g] + dd->alp[g]) * D0[2];

  args.E = exp(dd->phi[g] + dd->alp[g]) * D0[0];

  dd->del[g] = stepping_out_slice(dd, args);
}

void delSample(SEXP hh, chain_t *hd, chain_t *dd){
  if(!(vi(le(hh, "updates"), "del"))) return;
  del_kernel1<<<GRID, BLOCK>>>(dd);
}

#endif // DEL_H
