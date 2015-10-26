#include "utils/include.h"
#include "utils/element.h"
#include "utils/cuda_usage.h"
#include "utils/chain.h"
#include "utils/alloc_hd.h"
#include "utils/free_hd.h"
#include "utils/hd2hh.h"
#include "utils/hh2hd.h"
#include "utils/estimates.h"
#include "utils/eta.h"
#include "utils/priors.h"
#include "utils/curand_usage.h"
#include "utils/reset_starts.h"

#include "approx_gibbs_step/approx_gibbs_step_args.h"
#include "approx_gibbs_step/approx_gibbs_step_targets.h"
#include "approx_gibbs_step/stepping_out_slice.h"

#include "gibbs/phi.h"
#include "gibbs/alp.h"
#include "gibbs/del.h"
#include "gibbs/rho.h"
#include "gibbs/gam.h"
#include "gibbs/xiPhi.h"
#include "gibbs/xiAlp.h"
#include "gibbs/xiDel.h"
#include "gibbs/eps.h"
#include "gibbs/nuRho.h"
#include "gibbs/nuGam.h"
#include "gibbs/tauRho.h"
#include "gibbs/tauGam.h"
#include "gibbs/sigAlp.h"
#include "gibbs/sigDel.h"
#include "gibbs/sigPhi.h"
#include "gibbs/theAlp.h"
#include "gibbs/theDel.h"
#include "gibbs/thePhi.h"

void iteration(SEXP hh, chain_t *hd, chain_t *dd){
  epsSample(hh, hd, dd);
  rhoSample(hh, hd, dd);
  gamSample(hh, hd, dd);

  phiSample(hh, hd, dd);
  alpSample(hh, hd, dd);
  delSample(hh, hd, dd);

  xiPhiSample(hh, hd, dd);
  xiAlpSample(hh, hd, dd);
  xiDelSample(hh, hd, dd);

  nuRhoSample(hh, hd, dd);
  nuGamSample(hh, hd, dd);
  tauRhoSample(hh, hd, dd);
  tauGamSample(hh, hd, dd);

  thePhiSample(hh, hd, dd);
  theAlpSample(hh, hd, dd);
  theDelSample(hh, hd, dd);

  sigPhiSample(hh, hd, dd);
  sigAlpSample(hh, hd, dd);
  sigDelSample(hh, hd, dd);
}

void burnin(SEXP hh, chain_t *hd, chain_t *dd){
  int m,
      burnin = li(hh, "burnin")[0],
      print_every = burnin + 2,
      verbose = li(hh, "verbose")[0];

  if(!burnin) return;

  if(verbose){
    print_every = burnin/verbose + (burnin < verbose);
    Rprintf("Starting burnin on GPU %d.\n", getDevice());
  }

  for(m = 0; m < burnin; ++m){
    if(verbose && !((m + 1) % print_every))
      Rprintf("  burnin iteration %d of %d on GPU %d\n", m + 1, burnin, getDevice());

    iteration(hh, hd, dd);
  }
}

void chain(SEXP hh, chain_t *hd, chain_t *dd){
  int i, m,
      M = li(hh, "M")[0],
      print_every = M + 2,
      thin = li(hh, "thin")[0],
      verbose = li(hh, "verbose")[0];

  if(verbose){
    print_every = M/verbose + (M < verbose);
    Rprintf("Starting MCMC on GPU %d.\n", getDevice());
  }

  for(m = 0; m < M; ++m){
    if(verbose && !((m + 1) % print_every)){
      Rprintf("  MCMC iteration %d of %d on GPU %d", m + 1, M, getDevice());
      if(thin) Rprintf(" (thin = %d)", thin);
      Rprintf("\n");
    }

    iteration(hh, hd, dd);
    update_estimates(hh, dd);
    hd2hh(hh, hd, m);

    if(m < M - 1)
      for(i = 0; i < thin; ++i)
        iteration(hh, hd, dd);
  }
}

void end(SEXP hh, chain_t *hd, chain_t *dd){
  finish_estimates(hh, hd, dd);
  reset_starts(hh, hd);

  free_hd(hd);
  CUDA_CALL(cudaFree(dd));

  if(li(hh, "verbose")[0])
    Rprintf("Finished MCMC on GPU %d.\n", getDevice());

  cudaDeviceReset();
}

extern "C" SEXP heterosisCUDA(SEXP hh){
  if(li(hh, "verbose")[0])
    Rprintf("Loading MCMC on GPU %d.\n", getDevice());

  chain_t *hd = alloc_hd(hh);
  hh2hd(hh, hd);

  chain_t *dd;
  CUDA_CALL(cudaMalloc((void**) &dd, sizeof(chain_t)));
  CUDA_CALL(cudaMemcpy(dd, hd, sizeof(chain_t), cudaMemcpyHostToDevice));

  dim3 grid(GRID_N, GRID_G), block(BLOCK_N, BLOCK_G);
  curand_setup_kernel<<<grid, block>>>(dd);

  initialize_estimates(hh, dd);
  burnin(hh, hd, dd);
  chain(hh, hd, dd);
  end(hh, hd, dd);

  return hh;
}
